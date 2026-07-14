import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import glob
import math
import time
from math import radians, cos
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter1d
from main import (
    build_evaluation_samples, REFERENCE_DIR, TEST_DIR, OUTPUT_ROOT,
    OBS_WINDOW_LENGTH, PREDICTION_STEPS, N_TRAJECTORIES, device,
    haversine_km_vec, compute_displacement_errors,
    compute_ensemble_metrics, compute_mpqr,
    compute_heading_speed_error, classify_sample,
)
RESULT_CSV = os.path.join(OUTPUT_ROOT, "baseline_prob_gru_metrics.csv")
SUMMARY_TXT = os.path.join(OUTPUT_ROOT, "baseline_prob_gru_summary.txt")
MODEL_CACHE_PATH = os.path.join(OUTPUT_ROOT, "prob_gru_checkpoint.pt")
ACTIVITY_THRESHOLD_KM = 1.0
MATCH_PROPOSED_METHOD_SAMPLES = True
PROPOSED_METRICS_CSV = os.path.join(OUTPUT_ROOT, "per_sample_metrics.csv")
HIDDEN_SIZE = 64
NUM_EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
TRAIN_STRIDE = 5
FORCE_RETRAIN = False
VAR_MIN = 1e-6
VAR_MAX = 1e4
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
        raise FileNotFoundError(f"No data files were found in the reference directory: {reference_dir}")
    window = obs_len + pred_len
    obs_list = []
    fut_list = []
    for file_path in files:
        df = pd.read_csv(file_path)
        lons = df['LONGITUDE'].values
        lats = df['LATITUDE'].values
        num_points = len(lons)
        if num_points < window:
            continue
        filtered_lons = gaussian_filter1d(lons, sigma=1.0)
        filtered_lats = gaussian_filter1d(lats, sigma=1.0)
        for start_index in range(0, num_points - window + 1, stride):
            obs_lons = filtered_lons[start_index:start_index + obs_len]
            obs_lats = filtered_lats[start_index:start_index + obs_len]
            future_lons = lons[start_index + obs_len:start_index + window]
            future_lats = lats[start_index + obs_len:start_index + window]
            to_xy, _ = make_local_frame(obs_lons[-1], obs_lats[-1])
            obs_xy = np.array(
                [to_xy(lon, lat) for lon, lat in zip(obs_lons, obs_lats)],
                dtype=np.float32,
            )
            future_xy = np.array(
                [to_xy(lon, lat) for lon, lat in zip(future_lons, future_lats)],
                dtype=np.float32,
            )
            obs_list.append(obs_xy)
            fut_list.append(future_xy)
    if len(obs_list) == 0:
        raise RuntimeError(
            f"No valid training samples could be generated from {len(files)} reference trajectories."
        )
    print(
        f"[Training Data] Generated {len(obs_list)} training sample pairs "
        f"from {len(files)} reference trajectories using sliding windows "
        f"(observation steps: {obs_len}, prediction steps: {pred_len}, stride: {stride})."
    )
    return np.stack(obs_list), np.stack(fut_list)
class GlobalTrajDataset(Dataset):
    def __init__(self, obs_arr, fut_arr):
        self.obs = obs_arr
        self.fut = fut_arr
    def __len__(self):
        return len(self.obs)
    def __getitem__(self, idx):
        return torch.from_numpy(self.obs[idx]), torch.from_numpy(self.fut[idx])
class ProbEncoder(nn.Module):
    def __init__(self, input_size=2, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers=1,
            batch_first=True,
        )
    def forward(self, x):
        _, hidden = self.gru(x)
        return hidden.squeeze(0)
class ProbDecoder(nn.Module):
    def __init__(self, input_size=2, hidden_size=HIDDEN_SIZE, output_size=2):
        super().__init__()
        self.cell = nn.GRUCell(input_size, hidden_size)
        self.fc_mean = nn.Linear(hidden_size, output_size)
        self.fc_logvar = nn.Linear(hidden_size, output_size)
    def forward_step(self, prev, hidden):
        hidden = self.cell(prev, hidden)
        mean = self.fc_mean(hidden)
        variance = F.softplus(self.fc_logvar(hidden)) + 1e-6
        variance = torch.clamp(variance, min=VAR_MIN, max=VAR_MAX)
        return mean, variance, hidden
class ProbGRUModel(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.encoder = ProbEncoder(hidden_size=hidden_size)
        self.decoder = ProbDecoder(hidden_size=hidden_size)
    def forward_train_loss(self, obs_seq, target_seq):
        hidden = self.encoder(obs_seq)
        batch_size, num_steps, _ = target_seq.shape
        prev = torch.zeros(batch_size, 2, device=obs_seq.device)
        total_nll = torch.zeros((), device=obs_seq.device)
        for step_index in range(num_steps):
            mean, variance, hidden = self.decoder.forward_step(prev, hidden)
            total_nll = total_nll + 0.5 * torch.mean(
                torch.log(variance)
                + (target_seq[:, step_index, :] - mean) ** 2 / variance
            )
            prev = target_seq[:, step_index, :]
        return total_nll / num_steps
    @torch.no_grad()
    def sample_rollouts(self, obs_seq, n_steps, n_samples):
        initial_hidden = self.encoder(obs_seq)
        hidden = initial_hidden.repeat(n_samples, 1)
        prev = torch.zeros(n_samples, 2, device=obs_seq.device)
        all_xy = torch.zeros(
            n_samples,
            n_steps,
            2,
            device=obs_seq.device,
        )
        log_probs = torch.zeros(n_samples, device=obs_seq.device)
        for step_index in range(n_steps):
            mean, variance, hidden = self.decoder.forward_step(prev, hidden)
            standard_deviation = torch.sqrt(variance)
            noise = torch.randn_like(mean)
            sample = mean + noise * standard_deviation
            step_log_prob = -0.5 * (
                torch.log(2.0 * math.pi * variance)
                + (sample - mean) ** 2 / variance
            )
            log_probs = log_probs + step_log_prob.sum(dim=1)
            all_xy[:, step_index, :] = sample
            prev = sample
        return all_xy.cpu().numpy(), log_probs.cpu().numpy()
def train_or_load_model():
    if os.path.exists(MODEL_CACHE_PATH) and not FORCE_RETRAIN:
        print(
            f"[Model] A cached model was detected at {MODEL_CACHE_PATH}. "
            f"Loading it without retraining."
        )
        model = ProbGRUModel().to(device)
        checkpoint = torch.load(MODEL_CACHE_PATH, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        return model, 0.0
    print("[Model] No reusable cached model was detected. Starting one-time training...")
    obs_arr, fut_arr = build_training_pairs_from_reference(REFERENCE_DIR)
    dataset = GlobalTrajDataset(obs_arr, fut_arr)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    model = ProbGRUModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    training_start_time = time.time()
    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        num_batches = 0
        for obs_batch, fut_batch in loader:
            obs_batch = obs_batch.to(device)
            fut_batch = fut_batch.to(device)
            optimizer.zero_grad()
            loss = model.forward_train_loss(obs_batch, fut_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            average_loss = total_loss / max(num_batches, 1)
            print(
                f"  Epoch [{epoch + 1}/{NUM_EPOCHS}] Training loss "
                f"(Gaussian NLL in local kilometer coordinates): "
                f"{average_loss:.6f}"
            )
    training_time_sec = time.time() - training_start_time
    torch.save(model.state_dict(), MODEL_CACHE_PATH)
    print(
        f"[Model] Training completed in {training_time_sec:.1f} seconds. "
        f"The model was saved to {MODEL_CACHE_PATH}"
    )
    model.eval()
    return model, training_time_sec
def predict_prob_gru(model, obs_lons_filtered, obs_lats_filtered, n_trajectories=N_TRAJECTORIES):
    anchor_lon = obs_lons_filtered[-1]
    anchor_lat = obs_lats_filtered[-1]
    to_xy, to_lonlat = make_local_frame(anchor_lon, anchor_lat)
    obs_xy = np.array(
        [
            to_xy(lon, lat)
            for lon, lat in zip(obs_lons_filtered, obs_lats_filtered)
        ],
        dtype=np.float32,
    )
    obs_tensor = torch.from_numpy(obs_xy).unsqueeze(0).to(device)
    all_xy, log_probs = model.sample_rollouts(
        obs_tensor,
        PREDICTION_STEPS,
        n_trajectories,
    )
    all_pred_coords = []
    for trajectory_index in range(n_trajectories):
        pred_lons = [anchor_lon]
        pred_lats = [anchor_lat]
        for x, y in all_xy[trajectory_index]:
            lon, lat = to_lonlat(float(x), float(y))
            pred_lons.append(lon)
            pred_lats.append(lat)
        all_pred_coords.append((pred_lons, pred_lats))
    top_index = int(np.argmax(log_probs))
    return all_pred_coords, all_pred_coords[top_index]
def net_displacement_km(lons, lats):
    return haversine_km_vec(
        lats[0],
        lons[0],
        lats[-1],
        lons[-1],
    )
def print_and_log_summary(df, training_time_sec, lines_out):
    def write_line(text=''):
        lines_out.append(text)
        print(text)
    def write_block(subset, label):
        write_line(f"\n[{label}] Number of samples: {len(subset)}")
        metrics = [
            ('ADE_top1_km', 'ADE (Top-1)', 'km'),
            ('FDE_top1_km', 'FDE (Top-1)', 'km'),
            ('minADE_km', 'minADE (Oracle)', 'km'),
            ('minFDE_km', 'minFDE (Oracle)', 'km'),
            ('meanADE_km', 'meanADE (Ensemble Average)', 'km'),
            ('heading_MAE_deg', 'Heading MAE', 'degrees'),
            ('speed_MAE_kmh', 'Speed MAE', 'km/h'),
            ('MPQR95_km', 'MPQR@95 (Empirical Quantiles)', 'km'),
        ]
        for column, metric_label, unit in metrics:
            values = subset[column]
            write_line(
                f"  {metric_label}: Mean={values.mean():.3f} {unit}  "
                f"Median={values.median():.3f} {unit}  "
                f"Standard deviation={values.std():.3f}"
            )
        average_inference_ms = subset['infer_time_sec'].mean() * 1000.0
        write_line(
            f"  Average inference time per sample: "
            f"{average_inference_ms:.3f} ms"
        )
    write_line("=" * 78)
    write_line("Baseline: Probabilistic GRU (Prob-GRU) - Summary Results")
    write_line("=" * 78)
    write_line(
        f"Total one-time training duration: {training_time_sec:.1f} seconds "
        f"(a value of 0 indicates that a cached model was loaded without "
        f"retraining during this run)"
    )
    write_line(
        f"Number of Monte Carlo trajectory samples: {N_TRAJECTORIES} "
        f"(identical to the proposed model for comparable minADE and MPQR metrics)"
    )
    write_block(df, "All Samples")
    write_block(
        df[df['activity_category'] == 'Underway'],
        "Underway Subset",
    )
    write_line("=" * 78)
def main():
    print("=" * 78)
    print("Baseline: Probabilistic GRU (Prob-GRU)")
    print("=" * 78)
    model, training_time_sec = train_or_load_model()
    samples = build_evaluation_samples(TEST_DIR)
    if MATCH_PROPOSED_METHOD_SAMPLES and os.path.exists(PROPOSED_METRICS_CSV):
        proposed_df = pd.read_csv(
            PROPOSED_METRICS_CSV,
            encoding='utf-8-sig',
        )
        keep_ids = set(proposed_df['sample_id'].tolist())
        samples = [
            sample
            for sample in samples
            if sample['sample_id'] in keep_ids
        ]
        print(
            f"[Alignment] Detected {PROPOSED_METRICS_CSV}. Evaluating only "
            f"the {len(samples)} samples listed in that file to ensure that "
            f"the comparison with the proposed model uses exactly the same "
            f"test instances."
        )
    else:
        print(
            f"[Notice] Sample alignment is disabled or "
            f"{PROPOSED_METRICS_CSV} was not found. "
            f"Evaluating all {len(samples)} samples."
        )
    rows = []
    for sample in samples:
        inference_start_time = time.time()
        obs_lons_raw = sample['obs_lons']
        obs_lats_raw = sample['obs_lats']
        true_lons = sample['true_future_lons']
        true_lats = sample['true_future_lats']
        filtered_lons = gaussian_filter1d(obs_lons_raw, sigma=1.0)
        filtered_lats = gaussian_filter1d(obs_lats_raw, sigma=1.0)
        all_pred_coords, top1_coords = predict_prob_gru(
            model,
            filtered_lons,
            filtered_lats,
        )
        top1_ade, top1_fde, _ = compute_displacement_errors(
            top1_coords[0],
            top1_coords[1],
            true_lons,
            true_lats,
        )
        ensemble_metrics = compute_ensemble_metrics(
            all_pred_coords,
            true_lons,
            true_lats,
        )
        mpqr95, _ = compute_mpqr(
            all_pred_coords,
            true_lons,
            true_lats,
            confidence=0.95,
        )
        heading_mae, speed_mae = compute_heading_speed_error(
            top1_coords[0],
            top1_coords[1],
            true_lons,
            true_lats,
        )
        classification = classify_sample(
            true_lons,
            true_lats,
            filtered_lons[-1],
            filtered_lats[-1],
        )
        net_displacement = net_displacement_km(true_lons, true_lats)
        rows.append({
            'sample_id': sample['sample_id'],
            'source_file': sample['source_file'],
            'mmsi': sample['mmsi'],
            'ADE_top1_km': top1_ade,
            'FDE_top1_km': top1_fde,
            'minADE_km': ensemble_metrics['min_ade'],
            'minFDE_km': ensemble_metrics['min_fde'],
            'meanADE_km': ensemble_metrics['mean_ade'],
            'meanFDE_km': ensemble_metrics['mean_fde'],
            'MPQR95_km': mpqr95,
            'heading_MAE_deg': heading_mae,
            'speed_MAE_kmh': speed_mae,
            'net_displacement_km': net_displacement,
            'activity_category': (
                'Underway'
                if net_displacement >= ACTIVITY_THRESHOLD_KM
                else 'Near-stationary'
            ),
            'turn_category': classification['turn_category'],
            'speed_category': classification['speed_category'],
            'infer_time_sec': time.time() - inference_start_time,
        })
    df = pd.DataFrame(rows)
    df.to_csv(RESULT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n[Saved] {RESULT_CSV} ({len(df)} records)")
    lines = []
    print_and_log_summary(df, training_time_sec, lines)
    with open(SUMMARY_TXT, 'w', encoding='utf-8') as file:
        file.write('\n'.join(lines))
    print(f"\n[Saved] {SUMMARY_TXT}")
if __name__ == '__main__':
    main()

