import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from main import (
    TrajectoryModelCore,
    AngleDataset,
    SpeedDataset,
    SpeedLSTM,
    build_evaluation_samples,
    evaluate_sample,
    REFERENCE_DIR,
    TEST_DIR,
    OUTPUT_ROOT,
    N_MIXTURES,
    ANGLE_EPOCHS,
    SPEED_EPOCHS,
    OBS_WINDOW_LENGTH,
    PREDICTION_STEPS,

    haversine_km_vec,
)

RESULT_CSV = os.path.join(OUTPUT_ROOT, "ablation_no_attention_metrics.csv")
SUMMARY_TXT = os.path.join(OUTPUT_ROOT, "ablation_no_attention_summary.txt")
PROPOSED_METRICS_CSV = os.path.join(OUTPUT_ROOT, "per_sample_metrics.csv")

MATCH_PROPOSED_METHOD_SAMPLES = True
ACTIVITY_THRESHOLD_KM = 1.0

MAX_SPEED_TRAIN_SEC = 45
MAX_ANGLE_TRAIN_SEC = 120
ANGLE_EARLY_STOP_PATIENCE = 30
MIN_DELTA = 1e-4

SKIP_SAMPLE_IDS = set()
SKIP_SOURCE_FILES = set()
SKIP_MMSI = set()


class AngleMDN_NoAttention(nn.Module):
    def __init__(self, feature_dim=7, angle_dim=1,
                 hidden_dim=256, n_layers=3, n_mixtures=N_MIXTURES):
        super().__init__()
        self.n_mixtures = n_mixtures

        self.feature_lstm = nn.LSTM(
            feature_dim, hidden_dim, n_layers, batch_first=True, dropout=0
        )
        self.angle_lstm = nn.LSTM(
            angle_dim, hidden_dim // 2, n_layers, batch_first=True, dropout=0
        )

        total_hidden = hidden_dim + hidden_dim // 2
        self.batch_norm = nn.BatchNorm1d(total_hidden)

        self.mdn_output = nn.Sequential(
            nn.Linear(total_hidden, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(1e-38),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(1e-38),

            nn.Linear(128, n_mixtures * 3)
        )

    def forward(self, x_features, x_angle):
        feature_out, _ = self.feature_lstm(x_features)
        feature_summary = feature_out[:, -1, :]

        angle_out, _ = self.angle_lstm(x_angle)
        angle_summary = angle_out[:, -1, :]

        combined = torch.cat([feature_summary, angle_summary], dim=1)
        if combined.size(0) > 1:
            combined = self.batch_norm(combined)

        return self.mdn_output(combined)


class TrajectoryModelCoreNoAttention(TrajectoryModelCore):
    def train_speed_model(self, raw_speeds, epochs=SPEED_EPOCHS, lr=0.001,
                           max_train_sec=MAX_SPEED_TRAIN_SEC):
        local_speed_scaler = StandardScaler()
        speeds_scaled = local_speed_scaler.fit_transform(
            raw_speeds.reshape(-1, 1)
        ).flatten()

        dataset = SpeedDataset(speeds_scaled, self.sequence_length)
        if len(dataset) < 2:
            return None, local_speed_scaler

        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        model = SpeedLSTM().to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        model.train()
        train_t0 = time.time()

        for epoch in range(epochs):
            if max_train_sec is not None and time.time() - train_t0 > max_train_sec:
                break

            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()

        self.speed_model = model
        return model, local_speed_scaler

    def train_model(self, features, angles, epochs=ANGLE_EPOCHS, batch_size=64,
                     learning_rate=0.001, max_train_sec=MAX_ANGLE_TRAIN_SEC,
                     early_stop_patience=ANGLE_EARLY_STOP_PATIENCE,
                     min_delta=MIN_DELTA):

        angle_dataset = AngleDataset(features, angles, self.sequence_length)

        self.angle_model = AngleMDN_NoAttention(
            feature_dim=features.shape[1],
            n_mixtures=self.n_mixtures
        ).to(self.device)

        if len(angle_dataset) == 0:
            return [], []

        if len(angle_dataset) < 4:
            train_ds, val_ds = angle_dataset, angle_dataset
        else:
            train_size = int(0.8 * len(angle_dataset))
            val_size = len(angle_dataset) - train_size
            train_ds, val_ds = torch.utils.data.random_split(
                angle_dataset, [train_size, val_size]
            )

        train_loader = DataLoader(
            train_ds, batch_size=min(batch_size, max(1, len(train_ds))),
            shuffle=True, num_workers=0, drop_last=False
        )
        val_loader = DataLoader(
            val_ds, batch_size=min(batch_size, max(1, len(val_ds))),
            shuffle=False, num_workers=0
        )

        optimizer = optim.Adam(
            self.angle_model.parameters(), lr=learning_rate * 0.1, weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        train_losses, val_losses = [], []
        train_t0 = time.time()
        best_val = np.inf
        bad_epochs = 0

        for epoch in range(epochs):
            if max_train_sec is not None and time.time() - train_t0 > max_train_sec:
                break

            self.angle_model.train()
            train_loss = 0.0
            n_batches = 0

            for x_feat, x_ang, y_angle in train_loader:
                x_feat = x_feat.to(self.device)
                x_ang = x_ang.to(self.device)
                y_angle = y_angle.to(self.device)

                if torch.isnan(x_feat).any() or torch.isnan(x_ang).any() or torch.isnan(y_angle).any():
                    continue

                optimizer.zero_grad()
                mdn_params = self.angle_model(x_feat, x_ang)

                if torch.isnan(mdn_params).any() or torch.isinf(mdn_params).any():
                    continue

                loss = self.angle_mdn_loss(mdn_params, y_angle)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.angle_model.parameters(), max_norm=0.5)

                bad_grad = any(
                    p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                    for p in self.angle_model.parameters()
                )

                if not bad_grad:
                    optimizer.step()
                    train_loss += loss.item()
                    n_batches += 1

            if n_batches == 0:
                continue

            self.angle_model.eval()
            val_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for x_feat, x_ang, y_angle in val_loader:
                    x_feat = x_feat.to(self.device)
                    x_ang = x_ang.to(self.device)
                    y_angle = y_angle.to(self.device)

                    if torch.isnan(x_feat).any() or torch.isnan(x_ang).any():
                        continue

                    mdn_params = self.angle_model(x_feat, x_ang)
                    if torch.isnan(mdn_params).any() or torch.isinf(mdn_params).any():
                        continue

                    l = self.angle_mdn_loss(mdn_params, y_angle)
                    if torch.isnan(l) or torch.isinf(l):
                        continue

                    val_loss += l.item()
                    val_batches += 1

            if val_batches == 0:
                continue

            avg_train = train_loss / n_batches
            avg_val = val_loss / val_batches

            train_losses.append(avg_train)
            val_losses.append(avg_val)
            scheduler.step(avg_val)

            if avg_val < best_val - min_delta:
                best_val = avg_val
                bad_epochs = 0
            else:
                bad_epochs += 1

            if early_stop_patience is not None and bad_epochs >= early_stop_patience:
                break

        return train_losses, val_losses


_file_cache = {}


def add_activity_labels(df):
    net_disp_list = []

    for _, row in df.iterrows():
        sf = row['source_file']

        if sf not in _file_cache:
            tdf = pd.read_csv(os.path.join(TEST_DIR, sf))
            _file_cache[sf] = (tdf['LONGITUDE'].values, tdf['LATITUDE'].values)

        lons, lats = _file_cache[sf]
        s = int(row['window_start'])

        t_lons = lons[s + OBS_WINDOW_LENGTH: s + OBS_WINDOW_LENGTH + PREDICTION_STEPS]
        t_lats = lats[s + OBS_WINDOW_LENGTH: s + OBS_WINDOW_LENGTH + PREDICTION_STEPS]

        net_disp = haversine_km_vec(t_lats[0], t_lons[0], t_lats[-1], t_lons[-1])
        net_disp_list.append(net_disp)

    df = df.copy()
    df['net_displacement_km'] = net_disp_list
    df['activity_category'] = np.where(
        df['net_displacement_km'] >= ACTIVITY_THRESHOLD_KM,
        'Underway',
        'Near-stationary'
    )
    return df


def compare_block(sub_proposed, sub_ablation, label, lines_out):
    def W(s=''):
        lines_out.append(s)

    W(f"\n[{label}] Number of samples: {len(sub_ablation)}")
    W(f"{'Metric':<20}{'Full Model':>14}{'w/o Attention':>16}{'Rel. Change':>12}")

    metrics = [
        ('ADE_top1_km', 'ADE(top1,km)'),
        ('FDE_top1_km', 'FDE(top1,km)'),
        ('heading_MAE_deg', 'Heading MAE(deg)'),
        ('speed_MAE_kmh', 'Speed MAE(km/h)'),
        ('MPQR95_km', 'MPQR@95(km)'),
        ('minADE_km', 'minADE(km)'),
        ('minFDE_km', 'minFDE(km)'),
    ]

    for col, name in metrics:
        p_val = sub_proposed[col].mean()
        a_val = sub_ablation[col].mean()
        rel = (a_val - p_val) / p_val * 100 if abs(p_val) > 1e-9 else float('nan')
        W(f"{name:<20}{p_val:>14.3f}{a_val:>16.3f}{rel:>+11.1f}%")

    p_time = (sub_proposed['train_time_sec'] + sub_proposed['infer_time_sec']).mean()
    a_time = (sub_ablation['train_time_sec'] + sub_ablation['infer_time_sec']).mean()
    rel_t = (a_time - p_time) / p_time * 100 if abs(p_time) > 1e-9 else float('nan')

    W(f"{'Train+infer time(s)':<20}{p_time:>14.2f}{a_time:>16.2f}{rel_t:>+11.1f}%")


def compare_and_summarize(proposed_common, ablation_df):
    lines = []

    def W(s=''):
        lines.append(s)

    W("=" * 78)
    W("Ablation Experiment A: Full Model vs. w/o Attention")
    W("=" * 78)
    W(f"Common samples: {len(ablation_df)} (identical test instances as the full model)")

    compare_block(proposed_common, ablation_df, "All Samples", lines)

    underway_p = proposed_common[proposed_common['activity_category'] == 'Underway']
    underway_a = ablation_df[ablation_df['activity_category'] == 'Underway']
    compare_block(underway_p, underway_a, "Underway Subset", lines)

    W("=" * 78)
    return lines


def main():
    predictor = TrajectoryModelCoreNoAttention(reference_dir=REFERENCE_DIR)
    predictor.fit_scalers_from_reference()

    samples = build_evaluation_samples(TEST_DIR)

    if len(SKIP_SAMPLE_IDS) > 0 or len(SKIP_SOURCE_FILES) > 0 or len(SKIP_MMSI) > 0:
        skip_mmsi_str = {str(x) for x in SKIP_MMSI}
        samples = [
            s for s in samples
            if s['sample_id'] not in SKIP_SAMPLE_IDS
            and s['source_file'] not in SKIP_SOURCE_FILES
            and str(s['mmsi']) not in skip_mmsi_str
        ]

    proposed_df_full = None
    if MATCH_PROPOSED_METHOD_SAMPLES and os.path.exists(PROPOSED_METRICS_CSV):
        proposed_df_full = pd.read_csv(PROPOSED_METRICS_CSV, encoding='utf-8-sig')
        keep_ids = set(proposed_df_full['sample_id'].tolist())
        samples = [s for s in samples if s['sample_id'] in keep_ids]

    done_ids = set()
    if os.path.exists(RESULT_CSV):
        try:
            done_df = pd.read_csv(RESULT_CSV, encoding='utf-8-sig')
            done_ids = set(done_df['sample_id'].tolist())
        except Exception:
            os.remove(RESULT_CSV)

    remaining = [s for s in samples if s['sample_id'] not in done_ids]
    write_header = not os.path.exists(RESULT_CSV)

    for sample in remaining:
        try:
            row, _extra = evaluate_sample(predictor, sample)
        except Exception as e:
            failed_csv = os.path.join(OUTPUT_ROOT, "ablation_no_attention_failed_samples.csv")
            fail_row = {
                'sample_id': sample['sample_id'],
                'source_file': sample['source_file'],
                'mmsi': sample['mmsi'],
                'window_start': sample['window_start'],
                'error': str(e),
            }
            pd.DataFrame([fail_row]).to_csv(
                failed_csv, mode='a', index=False,
                header=not os.path.exists(failed_csv), encoding='utf-8-sig'
            )
            predictor.angle_model = None
            predictor.speed_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        pd.DataFrame([row]).to_csv(
            RESULT_CSV, mode='a', index=False,
            header=write_header, encoding='utf-8-sig'
        )
        write_header = False

        predictor.angle_model = None
        predictor.speed_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not os.path.exists(RESULT_CSV):
        return

    df = pd.read_csv(RESULT_CSV, encoding='utf-8-sig')

    if proposed_df_full is not None:
        df = add_activity_labels(df)

        proposed_common = proposed_df_full[
            proposed_df_full['sample_id'].isin(df['sample_id'])
        ].copy()
        proposed_common = add_activity_labels(proposed_common)

        lines = compare_and_summarize(proposed_common, df)

        with open(SUMMARY_TXT, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
