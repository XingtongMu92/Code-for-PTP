import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import glob
import time
from math import radians, cos
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter1d
from main import (
    build_evaluation_samples, REFERENCE_DIR, TEST_DIR, OUTPUT_ROOT,
    OBS_WINDOW_LENGTH, PREDICTION_STEPS, device, haversine_km_vec,
    compute_displacement_errors, compute_heading_speed_error, classify_sample,
)
RESULT_CSV = os.path.join(OUTPUT_ROOT, "baseline_global_lstm_metrics.csv")
SUMMARY_TXT = os.path.join(OUTPUT_ROOT, "baseline_global_lstm_summary.txt")
MODEL_CACHE_PATH = os.path.join(OUTPUT_ROOT, "global_lstm_checkpoint.pt")
ACTIVITY_THRESHOLD_KM = 1.0
MATCH_PROPOSED_METHOD_SAMPLES = True
PROPOSED_METRICS_CSV = os.path.join(OUTPUT_ROOT, "per_sample_metrics.csv")
HIDDEN_SIZE = 64
NUM_EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
TRAIN_STRIDE = 20
FORCE_RETRAIN = False
def make_local_frame(anchor_lon, anchor_lat):
    coslat0 = cos(radians(anchor_lat))
    def to_xy(lon, lat):
        return (lon - anchor_lon) * 111.0 * coslat0, (lat - anchor_lat) * 111.0
    def to_lonlat(x, y):
        return anchor_lon + x / (111.0 * coslat0), anchor_lat + y / 111.0
    return to_xy, to_lonlat
def build_training_pairs_from_reference(reference_dir, obs_len=OBS_WINDOW_LENGTH, pred_len=PREDICTION_STEPS, stride=TRAIN_STRIDE):
    files = sorted(glob.glob(os.path.join(reference_dir, '*_merged_resampled_5min.csv')))
    if len(files) == 0:
        raise FileNotFoundError(f'No data files were found in the reference directory: {reference_dir}')
    window = obs_len + pred_len
    obs_list, fut_list = [], []
    for f in files:
        df = pd.read_csv(f)
        lons, lats = df['LONGITUDE'].values, df['LATITUDE'].values
        n = len(lons)
        if n < window:
            continue
        filt_lons = gaussian_filter1d(lons, sigma=1.0)
        filt_lats = gaussian_filter1d(lats, sigma=1.0)
        for s in range(0, n - window + 1, stride):
            obs_lo = filt_lons[s:s + obs_len]
            obs_la = filt_lats[s:s + obs_len]
            fut_lo = lons[s + obs_len:s + window]
            fut_la = lats[s + obs_len:s + window]
            to_xy, _ = make_local_frame(obs_lo[-1], obs_la[-1])
            obs_xy = np.array([to_xy(lo, la) for lo, la in zip(obs_lo, obs_la)], dtype=np.float32)
            fut_xy = np.array([to_xy(lo, la) for lo, la in zip(fut_lo, fut_la)], dtype=np.float32)
            obs_list.append(obs_xy)
            fut_list.append(fut_xy)
    print(f"[Training Data] Generated {len(obs_list)} training sample pairs from {len(files)} reference trajectories using sliding windows (observation steps: {obs_len}, prediction steps: {pred_len}, stride: {stride})")
    return np.stack(obs_list), np.stack(fut_list)
class GlobalTrajDataset(Dataset):
    def __init__(self, obs_arr, fut_arr):
        self.obs = obs_arr
        self.fut = fut_arr
    def __len__(self):
        return len(self.obs)
    def __getitem__(self, idx):
        return torch.from_numpy(self.obs[idx]), torch.from_numpy(self.fut[idx])
class Encoder(nn.Module):
    def __init__(self, input_size=2, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
    def forward(self, x):
        _, (h, c) = self.lstm(x)
        return h.squeeze(0), c.squeeze(0)
class Decoder(nn.Module):
    def __init__(self, input_size=2, hidden_size=HIDDEN_SIZE, output_size=2):
        super().__init__()
        self.cell = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, h0, c0, n_steps, target_seq=None, teacher_forcing=True):
        batch_size = h0.size(0)
        h, c = h0, c0
        prev = torch.zeros(batch_size, 2, device=h0.device)
        outputs = []
        for t in range(n_steps):
            h, c = self.cell(prev, (h, c))
            pred = self.fc(h)
            outputs.append(pred)
            if teacher_forcing and target_seq is not None:
                prev = target_seq[:, t, :]
            else:
                prev = pred
        return torch.stack(outputs, dim=1)
class GlobalLSTMModel(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.encoder = Encoder(hidden_size=hidden_size)
        self.decoder = Decoder(hidden_size=hidden_size)
    def forward(self, obs_seq, n_steps=PREDICTION_STEPS, target_seq=None, teacher_forcing=True):
        h0, c0 = self.encoder(obs_seq)
        return self.decoder(h0, c0, n_steps, target_seq=target_seq, teacher_forcing=teacher_forcing)
def train_or_load_model():
    if os.path.exists(MODEL_CACHE_PATH) and not FORCE_RETRAIN:
        print(f"[Model] A cached model was detected at {MODEL_CACHE_PATH}. Loading it without retraining.")
        model = GlobalLSTMModel().to(device)
        model.load_state_dict(torch.load(MODEL_CACHE_PATH, map_location=device))
        model.eval()
        return model, 0.0
    print("[Model] No reusable cached model was detected. Starting one-time training...")
    obs_arr, fut_arr = build_training_pairs_from_reference(REFERENCE_DIR)
    dataset = GlobalTrajDataset(obs_arr, fut_arr)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    model = GlobalLSTMModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    t_train_start = time.time()
    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss, n_batches = 0.0, 0
        for obs_batch, fut_batch in loader:
            obs_batch = obs_batch.to(device)
            fut_batch = fut_batch.to(device)
            optimizer.zero_grad()
            pred = model(obs_batch, n_steps=fut_batch.size(1), target_seq=fut_batch, teacher_forcing=True)
            loss = criterion(pred, fut_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch [{epoch + 1}/{NUM_EPOCHS}] Training loss (MSE in local kilometer coordinates): {total_loss / max(n_batches, 1):.6f}")
    training_time_sec = time.time() - t_train_start
    torch.save(model.state_dict(), MODEL_CACHE_PATH)
    print(f"[Model] Training completed in {training_time_sec:.1f} seconds. The model was saved to {MODEL_CACHE_PATH}")
    model.eval()
    return model, training_time_sec
def predict_global_lstm(model, obs_lons_filtered, obs_lats_filtered):
    anchor_lon, anchor_lat = obs_lons_filtered[-1], obs_lats_filtered[-1]
    to_xy, to_lonlat = make_local_frame(anchor_lon, anchor_lat)
    obs_xy = np.array([to_xy(lo, la) for lo, la in zip(obs_lons_filtered, obs_lats_filtered)], dtype=np.float32)
    obs_tensor = torch.from_numpy(obs_xy).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_xy = model(obs_tensor, n_steps=PREDICTION_STEPS, target_seq=None, teacher_forcing=False).cpu().numpy()[0]
    pred_lons, pred_lats = [anchor_lon], [anchor_lat]
    for x, y in pred_xy:
        lo, la = to_lonlat(float(x), float(y))
        pred_lons.append(lo)
        pred_lats.append(la)
    return pred_lons, pred_lats
def net_displacement_km(lons, lats):
    return haversine_km_vec(lats[0], lons[0], lats[-1], lons[-1])
def print_and_log_summary(df, training_time_sec, lines_out):
    def write_line(s=''):
        lines_out.append(s)
        print(s)
    def block(sub, label):
        write_line(f"\n[{label}] Number of samples: {len(sub)}")
        for col, metric_label, unit in [('ADE_km', 'ADE', 'km'), ('FDE_km', 'FDE', 'km'), ('heading_MAE_deg', 'Heading MAE', 'degrees'), ('speed_MAE_kmh', 'Speed MAE', 'km/h')]:
            values = sub[col]
            write_line(f"  {metric_label}: Mean={values.mean():.3f} {unit}  Median={values.median():.3f} {unit}  Standard deviation={values.std():.3f}")
        write_line(f"  Average inference time per sample: {sub['infer_time_sec'].mean() * 1000:.3f} ms")
    write_line("=" * 78)
    write_line("Baseline: Global Sequence-to-Sequence LSTM - Summary Results")
    write_line("=" * 78)
    write_line(f"Total one-time training duration: {training_time_sec:.1f} seconds (a value of 0 indicates that a cached model was loaded without retraining during this run)")
    block(df, "All Samples")
    block(df[df['activity_category'] == 'Underway'], "Underway Subset")
    write_line("=" * 78)
def main():
    print("=" * 78)
    print("Baseline: Global Sequence-to-Sequence LSTM")
    print("=" * 78)
    model, training_time_sec = train_or_load_model()
    samples = build_evaluation_samples(TEST_DIR)
    if MATCH_PROPOSED_METHOD_SAMPLES and os.path.exists(PROPOSED_METRICS_CSV):
        proposed_df = pd.read_csv(PROPOSED_METRICS_CSV, encoding='utf-8-sig')
        keep_ids = set(proposed_df['sample_id'].tolist())
        samples = [s for s in samples if s['sample_id'] in keep_ids]
        print(f"[Alignment] Detected {PROPOSED_METRICS_CSV}. Evaluating only the {len(samples)} samples listed in that file to ensure that the comparison with the proposed model uses exactly the same test instances.")
    else:
        print(f"[Notice] Sample alignment is disabled or {PROPOSED_METRICS_CSV} was not found. Evaluating all {len(samples)} samples.")
    rows = []
    for sample in samples:
        t0 = time.time()
        obs_lons_raw = sample['obs_lons']
        obs_lats_raw = sample['obs_lats']
        true_lons = sample['true_future_lons']
        true_lats = sample['true_future_lats']
        filtered_lons = gaussian_filter1d(obs_lons_raw, sigma=1.0)
        filtered_lats = gaussian_filter1d(obs_lats_raw, sigma=1.0)
        pred_lons, pred_lats = predict_global_lstm(model, filtered_lons, filtered_lats)
        ade, fde, _ = compute_displacement_errors(pred_lons, pred_lats, true_lons, true_lats)
        heading_mae, speed_mae = compute_heading_speed_error(pred_lons, pred_lats, true_lons, true_lats)
        cls_info = classify_sample(true_lons, true_lats, filtered_lons[-1], filtered_lats[-1])
        net_disp = net_displacement_km(true_lons, true_lats)
        rows.append({
            'sample_id': sample['sample_id'],
            'source_file': sample['source_file'],
            'mmsi': sample['mmsi'],
            'ADE_km': ade,
            'FDE_km': fde,
            'heading_MAE_deg': heading_mae,
            'speed_MAE_kmh': speed_mae,
            'net_displacement_km': net_disp,
            'activity_category': 'Underway' if net_disp >= ACTIVITY_THRESHOLD_KM else 'Near-stationary',
            'turn_category': cls_info['turn_category'],
            'speed_category': cls_info['speed_category'],
            'infer_time_sec': time.time() - t0,
        })
    df = pd.DataFrame(rows)
    df.to_csv(RESULT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n[Saved] {RESULT_CSV} ({len(df)} records)")
    lines = []
    print_and_log_summary(df, training_time_sec, lines)
    with open(SUMMARY_TXT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\n[Saved] {SUMMARY_TXT}")
if __name__ == '__main__':
    main()
