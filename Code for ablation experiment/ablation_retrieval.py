import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import gc
import pickle
import time
import concurrent.futures

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import StandardScaler

from main import (
    build_evaluation_samples, REFERENCE_DIR, TEST_DIR, OUTPUT_ROOT,
    N_TRAJECTORIES, N_MIXTURES, device,
    compute_displacement_errors, compute_ensemble_metrics,
    compute_mpqr, compute_heading_speed_error, classify_sample,
    TrajectoryModelCore, AngleMDN, SpeedLSTM, AngleDataset, SpeedDataset,
)

RESULT_CSV = os.path.join(OUTPUT_ROOT, "global_dual_lstm_attn_mdn_metrics.csv")
SUMMARY_TXT = os.path.join(OUTPUT_ROOT, "global_dual_lstm_attn_mdn_summary.txt")
CHECKPOINT_PATH = os.path.join(OUTPUT_ROOT, "global_dual_lstm_attn_mdn_checkpoint.pt")
SPEED_SCALER_PATH = os.path.join(OUTPUT_ROOT, "global_dual_lstm_attn_mdn_speed_scaler.pkl")

MATCH_PROPOSED_METHOD_SAMPLES = True
PROPOSED_METRICS_CSV = os.path.join(OUTPUT_ROOT, "per_sample_metrics.csv")

FORCE_RETRAIN = False

GLOBAL_ANGLE_EPOCHS = 300
GLOBAL_SPEED_EPOCHS = 300
GLOBAL_EARLY_STOP_PATIENCE = 15
GLOBAL_MAX_TRAIN_SEC = 1800

PER_SAMPLE_INFER_TIME_CAP_SEC = 300


def _run_with_timeout(func, args, timeout_sec):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args)
        return future.result(timeout=timeout_sec)


def _train_global_angle_mdn(predictor, features, angles, epochs, batch_size=64,
                             learning_rate=0.001, early_stop_patience=GLOBAL_EARLY_STOP_PATIENCE,
                             max_train_sec=GLOBAL_MAX_TRAIN_SEC):
    angle_dataset = AngleDataset(features, angles, predictor.sequence_length)
    train_size = int(0.8 * len(angle_dataset))
    val_size = len(angle_dataset) - train_size
    train_ds, val_ds = random_split(angle_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    angle_model = AngleMDN(feature_dim=features.shape[1], n_mixtures=predictor.n_mixtures).to(device)
    predictor.angle_model = angle_model
    optimizer = optim.Adam(angle_model.parameters(), lr=learning_rate * 0.1, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val, bad_epochs = np.inf, 0
    t0 = time.time()

    for epoch in range(epochs):
        if time.time() - t0 > max_train_sec:
            break

        angle_model.train()
        train_loss, n_batches = 0.0, 0
        for x_feat, x_ang, y_angle in train_loader:
            x_feat, x_ang, y_angle = x_feat.to(device), x_ang.to(device), y_angle.to(device)
            optimizer.zero_grad()
            mdn_params = angle_model(x_feat, x_ang)
            loss = predictor.angle_mdn_loss(mdn_params, y_angle)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(angle_model.parameters(), max_norm=0.5)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        if n_batches == 0:
            continue

        angle_model.eval()
        val_loss, val_batches = 0.0, 0
        with torch.no_grad():
            for x_feat, x_ang, y_angle in val_loader:
                x_feat, x_ang, y_angle = x_feat.to(device), x_ang.to(device), y_angle.to(device)
                mdn_params = angle_model(x_feat, x_ang)
                l = predictor.angle_mdn_loss(mdn_params, y_angle)
                if torch.isnan(l) or torch.isinf(l):
                    continue
                val_loss += l.item()
                val_batches += 1
        if val_batches == 0:
            continue

        avg_val = val_loss / val_batches
        scheduler.step(avg_val)

        if avg_val < best_val - 1e-4:
            best_val, bad_epochs = avg_val, 0
        else:
            bad_epochs += 1

        if early_stop_patience is not None and bad_epochs >= early_stop_patience:
            break

    angle_model.eval()
    return angle_model


def _train_global_speed_lstm(predictor, raw_speeds, epochs, batch_size=32, lr=0.001,
                              early_stop_patience=GLOBAL_EARLY_STOP_PATIENCE,
                              max_train_sec=GLOBAL_MAX_TRAIN_SEC):
    local_speed_scaler = StandardScaler()
    speeds_scaled = local_speed_scaler.fit_transform(raw_speeds.reshape(-1, 1)).flatten()

    dataset = SpeedDataset(speeds_scaled, predictor.sequence_length)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = SpeedLSTM().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val, bad_epochs = np.inf, 0
    t0 = time.time()

    for epoch in range(epochs):
        if time.time() - t0 > max_train_sec:
            break

        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss, val_batches = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item()
                val_batches += 1
        if val_batches == 0:
            continue
        avg_val = val_loss / val_batches

        if avg_val < best_val - 1e-5:
            best_val, bad_epochs = avg_val, 0
        else:
            bad_epochs += 1

        if early_stop_patience is not None and bad_epochs >= early_stop_patience:
            break

    model.eval()
    return model, local_speed_scaler


def train_or_load_global_models(predictor):
    if os.path.exists(CHECKPOINT_PATH) and os.path.exists(SPEED_SCALER_PATH) and not FORCE_RETRAIN:
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)

        angle_model = AngleMDN(feature_dim=7, n_mixtures=N_MIXTURES).to(device)
        angle_model.load_state_dict(ckpt['angle_model_state_dict'])
        angle_model.eval()

        speed_model = SpeedLSTM().to(device)
        speed_model.load_state_dict(ckpt['speed_model_state_dict'])
        speed_model.eval()

        with open(SPEED_SCALER_PATH, 'rb') as f:
            speed_scaler = pickle.load(f)

        return angle_model, speed_model, speed_scaler, 0.0

    t0 = time.time()

    angle_model = _train_global_angle_mdn(
        predictor, predictor.pooled_features_scaled, predictor.pooled_angles_scaled,
        epochs=GLOBAL_ANGLE_EPOCHS,
    )
    predictor.angle_model = angle_model

    pooled_speed_scaled_col = predictor.pooled_features_scaled[:, 4].reshape(-1, 1)
    pooled_raw_speeds = predictor.speed_scaler.inverse_transform(pooled_speed_scaled_col).flatten()
    speed_model, speed_scaler = _train_global_speed_lstm(
        predictor, pooled_raw_speeds, epochs=GLOBAL_SPEED_EPOCHS,
    )

    training_time_sec = time.time() - t0

    torch.save({
        'angle_model_state_dict': angle_model.state_dict(),
        'speed_model_state_dict': speed_model.state_dict(),
    }, CHECKPOINT_PATH)
    with open(SPEED_SCALER_PATH, 'wb') as f:
        pickle.dump(speed_scaler, f)

    return angle_model, speed_model, speed_scaler, training_time_sec


def evaluate_sample_global(predictor, sample, speed_model, speed_scaler):
    t0 = time.time()
    obs_lons_raw, obs_lats_raw, obs_times = sample['obs_lons'], sample['obs_lats'], sample['obs_times']
    true_lons, true_lats = sample['true_future_lons'], sample['true_future_lats']

    filtered_lons, filtered_lats = predictor.filter_trajectory(obs_lons_raw, obs_lats_raw)
    features, headings, turning_angles = predictor.feature_engineer.calculate_features(
        filtered_lats, filtered_lons, obs_times)
    turning_angles_corrected = predictor.correct_turning_angles(features, turning_angles)

    features_scaled = predictor.feature_scaler.transform(features)
    angles_scaled = predictor.angle_scaler.transform(turning_angles_corrected.reshape(-1, 1)).flatten()
    speeds_scaled = predictor.speed_scaler.transform(features[:, 4].reshape(-1, 1)).flatten()

    seq_len = predictor.sequence_length
    start_idx = max(0, len(filtered_lons) - seq_len)
    init_feat = features_scaled[start_idx:]
    init_ang = angles_scaled[start_idx:]
    init_spd = speeds_scaled[start_idx:]
    init_head = headings[start_idx:]
    init_lon = filtered_lons[start_idx:]
    init_lat = filtered_lats[start_idx:]

    t_train_end = time.time()

    all_trajs_angle, all_trajs_speed, all_probs = predictor.predict_trajectories(
        init_feat, init_ang, init_spd, init_head, init_lon, init_lat,
        n_trajectories=N_TRAJECTORIES,
        speed_model_components=(speed_model, speed_scaler),
    )

    all_pred_coords = [predictor.reconstruct_trajectory(init_lon, init_lat, init_head[-1], a, s)
                       for a, s in zip(all_trajs_angle, all_trajs_speed)]
    t_infer_end = time.time()

    top_idx = np.argsort(all_probs)[-5:][::-1]
    top5_coords = [all_pred_coords[i] for i in top_idx]
    top5_probs = [all_probs[i] for i in top_idx]

    ens_metrics = compute_ensemble_metrics(all_pred_coords, true_lons, true_lats)
    top1_ade, top1_fde, _ = compute_displacement_errors(top5_coords[0][0], top5_coords[0][1], true_lons, true_lats)
    mpqr95, _ = compute_mpqr(all_pred_coords, true_lons, true_lats, confidence=0.95)
    heading_mae, speed_mae = compute_heading_speed_error(top5_coords[0][0], top5_coords[0][1], true_lons, true_lats)
    cls_info = classify_sample(true_lons, true_lats, init_lon[-1], init_lat[-1])

    result_row = {
        'sample_id': sample['sample_id'], 'source_file': sample['source_file'],
        'mmsi': sample['mmsi'],
        'window_start': sample['window_start'],
        'n_similar_trajectories': -1,
        'ADE_top1_km': top1_ade, 'FDE_top1_km': top1_fde,
        'minADE_km': ens_metrics['min_ade'], 'minFDE_km': ens_metrics['min_fde'],
        'meanADE_km': ens_metrics['mean_ade'], 'meanFDE_km': ens_metrics['mean_fde'],
        'MPQR95_km': mpqr95,
        'heading_MAE_deg': heading_mae, 'speed_MAE_kmh': speed_mae,
        'cum_turn_deg': cls_info['cum_turn_deg'], 'speed_cv': cls_info['speed_cv'],
        'turn_category': cls_info['turn_category'], 'speed_category': cls_info['speed_category'],
        'train_time_sec': t_train_end - t0,
        'infer_time_sec': t_infer_end - t_train_end,
    }
    return result_row


def print_and_log_summary(df, global_training_time_sec, lines_out):
    def W(s=''):
        lines_out.append(s)

    def block(sub, label):
        W(f"\n[{label}] Number of samples: {len(sub)}")
        for col, metric_label, unit in [
            ('ADE_top1_km', 'ADE(Top-1)', 'km'), ('FDE_top1_km', 'FDE(Top-1)', 'km'),
            ('minADE_km', 'minADE(oracle)', 'km'), ('minFDE_km', 'minFDE(oracle)', 'km'),
            ('meanADE_km', 'meanADE(ensemble average)', 'km'),
            ('heading_MAE_deg', 'Heading MAE', 'deg'), ('speed_MAE_kmh', 'Speed MAE', 'km/h'),
            ('MPQR95_km', 'MPQR@95(empirical quantile)', 'km'),
        ]:
            v = sub[col]
            W(f"  {metric_label}: mean={v.mean():.3f}{unit}  median={v.median():.3f}{unit}  std={v.std():.3f}")
        W(f"  Average inference time per sample: {sub['infer_time_sec'].mean() * 1000:.3f} ms")

    W("=" * 78)
    W("Ablation: Global Dual-LSTM-Attn-MDN - Summary Results")
    W("=" * 78)
    W(f"Total one-time global training duration: {global_training_time_sec:.1f} seconds "
      f"(a value of 0 indicates that a cached model was loaded without retraining during this run)")
    W(f"Number of Monte Carlo trajectory samples: {N_TRAJECTORIES} "
      f"(identical to the proposed model for comparable minADE/MPQR)")
    W("Architecture identical to the proposed model (AngleMDN + SpeedLSTM); the only "
      "difference is that this ablation is trained once on the entire reference "
      "library and performs no retrieval or per-sample retraining during evaluation.")
    block(df, "All Samples")
    W("=" * 78)


def main():
    predictor = TrajectoryModelCore(reference_dir=REFERENCE_DIR)
    predictor.fit_scalers_from_reference()

    angle_model, speed_model, speed_scaler, global_training_time_sec = train_or_load_global_models(predictor)
    predictor.angle_model = angle_model

    samples = build_evaluation_samples(TEST_DIR)

    if MATCH_PROPOSED_METHOD_SAMPLES and os.path.exists(PROPOSED_METRICS_CSV):
        proposed_df = pd.read_csv(PROPOSED_METRICS_CSV, encoding='utf-8-sig')
        keep_ids = set(proposed_df['sample_id'].tolist())
        samples = [s for s in samples if s['sample_id'] in keep_ids]

    done_ids = set()
    if os.path.exists(RESULT_CSV):
        try:
            done_df = pd.read_csv(RESULT_CSV, encoding='utf-8-sig')
            done_ids = set(done_df['sample_id'].tolist())
        except Exception:
            os.remove(RESULT_CSV)

    remaining_samples = [s for s in samples if s['sample_id'] not in done_ids]
    write_header = not os.path.exists(RESULT_CSV)

    for i, sample in enumerate(remaining_samples):
        try:
            row = _run_with_timeout(
                evaluate_sample_global, (predictor, sample, speed_model, speed_scaler),
                timeout_sec=PER_SAMPLE_INFER_TIME_CAP_SEC,
            )
        except concurrent.futures.TimeoutError:
            continue
        except Exception:
            continue

        pd.DataFrame([row]).to_csv(RESULT_CSV, mode='a', index=False,
                                    header=write_header, encoding='utf-8-sig')
        write_header = False

        if torch.cuda.is_available() and (i + 1) % 50 == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()

    df = pd.read_csv(RESULT_CSV, encoding='utf-8-sig')

    lines = []
    print_and_log_summary(df, global_training_time_sec, lines)
    with open(SUMMARY_TXT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
