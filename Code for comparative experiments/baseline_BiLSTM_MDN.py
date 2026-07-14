import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import glob
import time
from math import radians, cos, pi
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter1d
from main import (
    build_evaluation_samples, REFERENCE_DIR, TEST_DIR, OUTPUT_ROOT,
    OBS_WINDOW_LENGTH, PREDICTION_STEPS, N_TRAJECTORIES, device,
    haversine_km_vec, compute_displacement_errors,
    compute_ensemble_metrics, compute_heading_speed_error, classify_sample,
    compute_mpqr,
)
RESULT_CSV = os.path.join(OUTPUT_ROOT, "baseline_global_mdn_bilstm_metrics.csv")
SUMMARY_TXT = os.path.join(OUTPUT_ROOT, "baseline_global_mdn_bilstm_summary.txt")
MODEL_CACHE_PATH = os.path.join(OUTPUT_ROOT, "global_mdn_bilstm_checkpoint.pt")
ACTIVITY_THRESHOLD_KM = 1.0
MATCH_PROPOSED_METHOD_SAMPLES = True
PROPOSED_METRICS_CSV = os.path.join(OUTPUT_ROOT, "per_sample_metrics.csv")
HIDDEN_SIZE = 64
NUM_LAYERS = 1
N_MIXTURES = 11
NUM_EPOCHS = 120
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
TRAIN_STRIDE = 20
GRAD_CLIP = 1.0
FORCE_RETRAIN = False
GLOBAL_RANDOM_SEED = 42
MIN_SIGMA = 1e-3
LOG_SIGMA_MIN = -5.0
LOG_SIGMA_MAX = 3.0
def make_local_frame(anchor_lon, anchor_lat):
    coslat0 = cos(radians(anchor_lat))
    coslat0 = max(coslat0, 1e-6)
    def to_xy(lon, lat):
        return (lon - anchor_lon) * 111.0 * coslat0, (lat - anchor_lat) * 111.0
    def to_lonlat(x, y):
        return anchor_lon + x / (111.0 * coslat0), anchor_lat + y / 111.0
    return to_xy, to_lonlat
def build_training_pairs_from_reference(reference_dir, obs_len=OBS_WINDOW_LENGTH, pred_len=PREDICTION_STEPS, stride=TRAIN_STRIDE):
    files = sorted(glob.glob(os.path.join(reference_dir, '*_merged_resampled_5min.csv')))
    if len(files) == 0:
        files = sorted(glob.glob(os.path.join(reference_dir, '*.csv')))
    if len(files) == 0:
        raise FileNotFoundError(f"No CSV data files were found in the reference directory: {reference_dir}")
    window = obs_len + pred_len
    obs_list, fut_list = [], []
    for file_path in files:
        df = pd.read_csv(file_path)
        if 'LONGITUDE' not in df.columns or 'LATITUDE' not in df.columns:
            print(f"[Warning] The file is missing the LONGITUDE or LATITUDE column and was skipped: {file_path}")
            continue
        lons = df['LONGITUDE'].values
        lats = df['LATITUDE'].values
        n = len(lons)
        if n < window:
            continue
        filtered_lons = gaussian_filter1d(lons, sigma=1.0)
        filtered_lats = gaussian_filter1d(lats, sigma=1.0)
        for start in range(0, n - window + 1, stride):
            obs_lons = filtered_lons[start:start + obs_len]
            obs_lats = filtered_lats[start:start + obs_len]
            future_lons = lons[start + obs_len:start + window]
            future_lats = lats[start + obs_len:start + window]
            anchor_lon = obs_lons[-1]
            anchor_lat = obs_lats[-1]
            to_xy, _ = make_local_frame(anchor_lon, anchor_lat)
            obs_xy = np.array(
                [to_xy(lon, lat) for lon, lat in zip(obs_lons, obs_lats)],
                dtype=np.float32
            )
            future_xy = np.array(
                [to_xy(lon, lat) for lon, lat in zip(future_lons, future_lats)],
                dtype=np.float32
            )
            obs_list.append(obs_xy)
            fut_list.append(future_xy)
    if len(obs_list) == 0:
        raise RuntimeError("No training samples could be constructed from the reference trajectories. Check the trajectory lengths and window parameters.")
    print(
        f"[Training Data] Generated {len(obs_list)} training sample pairs from "
        f"{len(files)} reference trajectories using sliding windows "
        f"(observation steps: {obs_len}, prediction steps: {pred_len}, stride: {stride})"
    )
    return np.stack(obs_list), np.stack(fut_list)
class GlobalMDNDataset(Dataset):
    def __init__(self, obs_arr, fut_arr):
        self.obs = obs_arr
        self.fut = fut_arr
    def __len__(self):
        return len(self.obs)
    def __getitem__(self, idx):
        return torch.from_numpy(self.obs[idx]), torch.from_numpy(self.fut[idx])
class GlobalMDNBiLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=HIDDEN_SIZE, n_layers=NUM_LAYERS, n_mixtures=N_MIXTURES, pred_len=PREDICTION_STEPS):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_mixtures = n_mixtures
        self.pred_len = pred_len
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True
        )
        encoder_dim = hidden_size * 2
        self.head = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, pred_len * n_mixtures * 5)
        )
    def forward(self, obs_seq):
        output, _ = self.encoder(obs_seq)
        last_hidden = output[:, -1, :]
        raw = self.head(last_hidden)
        raw = raw.view(-1, self.pred_len, self.n_mixtures, 5)
        logits = raw[..., 0]
        mu = raw[..., 1:3]
        log_sigma = raw[..., 3:5].clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX)
        sigma = torch.exp(log_sigma) + MIN_SIGMA
        return logits, mu, sigma
def mdn_nll_loss(logits, mu, sigma, target):
    expanded_target = target.unsqueeze(2)
    log_pi = torch.log_softmax(logits, dim=-1)
    normalized = (expanded_target - mu) / sigma
    log_determinant = torch.log(sigma[..., 0]) + torch.log(sigma[..., 1])
    quadratic_term = 0.5 * torch.sum(normalized ** 2, dim=-1)
    log_probability = -np.log(2.0 * np.pi) - log_determinant - quadratic_term
    mixture_log_probability = torch.logsumexp(log_pi + log_probability, dim=-1)
    return -mixture_log_probability.mean()
def train_or_load_model():
    if os.path.exists(MODEL_CACHE_PATH) and not FORCE_RETRAIN:
        print(f"[Model] A cached model was detected at {MODEL_CACHE_PATH}. Loading it without retraining.")
        model = GlobalMDNBiLSTM().to(device)
        model.load_state_dict(torch.load(MODEL_CACHE_PATH, map_location=device))
        model.eval()
        return model, 0.0
    print("[Model] No reusable cached model was detected or FORCE_RETRAIN is enabled. Starting one-time Global MDN-BiLSTM training...")
    obs_arr, fut_arr = build_training_pairs_from_reference(REFERENCE_DIR)
    dataset = GlobalMDNDataset(obs_arr, fut_arr)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    model = GlobalMDNBiLSTM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    training_start_time = time.time()
    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        number_of_batches = 0
        for obs_batch, fut_batch in loader:
            obs_batch = obs_batch.to(device)
            fut_batch = fut_batch.to(device)
            optimizer.zero_grad()
            logits, mu, sigma = model(obs_batch)
            loss = mdn_nll_loss(logits, mu, sigma, fut_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total_loss += loss.item()
            number_of_batches += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  Epoch [{epoch + 1}/{NUM_EPOCHS}] "
                f"NLL={total_loss / max(number_of_batches, 1):.6f}"
            )
    training_time_sec = time.time() - training_start_time
    torch.save(model.state_dict(), MODEL_CACHE_PATH)
    print(
        f"[Model] Training completed in {training_time_sec:.1f} seconds. "
        f"The model was saved to {MODEL_CACHE_PATH}"
    )
    model.eval()
    return model, training_time_sec
def gaussian_log_prob_2d(xy, mu, sigma):
    dx = (xy[0] - mu[0]) / sigma[0]
    dy = (xy[1] - mu[1]) / sigma[1]
    return (
        -np.log(2.0 * pi)
        - np.log(sigma[0])
        - np.log(sigma[1])
        - 0.5 * (dx * dx + dy * dy)
    )
def mixture_log_prob_2d(xy, logits_t, mu_t, sigma_t):
    normalized_logits = logits_t - np.max(logits_t)
    mixture_weights = np.exp(normalized_logits)
    mixture_weights = mixture_weights / np.sum(mixture_weights)
    values = []
    for mixture_index in range(len(mixture_weights)):
        values.append(
            np.log(mixture_weights[mixture_index] + 1e-12)
            + gaussian_log_prob_2d(
                xy,
                mu_t[mixture_index],
                sigma_t[mixture_index]
            )
        )
    values = np.array(values)
    maximum_value = np.max(values)
    return float(
        maximum_value
        + np.log(np.sum(np.exp(values - maximum_value)) + 1e-12)
    )
def sample_mdn_trajectories(logits, mu, sigma, n_trajectories=N_TRAJECTORIES, random_seed=GLOBAL_RANDOM_SEED):
    rng = np.random.default_rng(random_seed)
    prediction_length, number_of_mixtures = logits.shape
    sampled_trajectories = []
    log_scores = []
    for _ in range(n_trajectories):
        trajectory = []
        score = 0.0
        for step in range(prediction_length):
            probabilities = np.exp(logits[step] - np.max(logits[step]))
            probabilities = probabilities / np.sum(probabilities)
            mixture_index = int(
                rng.choice(number_of_mixtures, p=probabilities)
            )
            xy = rng.normal(
                loc=mu[step, mixture_index],
                scale=sigma[step, mixture_index]
            ).astype(np.float32)
            trajectory.append(xy)
            score += mixture_log_prob_2d(
                xy,
                logits[step],
                mu[step],
                sigma[step]
            )
        sampled_trajectories.append(
            np.array(trajectory, dtype=np.float32)
        )
        log_scores.append(score)
    return sampled_trajectories, np.array(log_scores, dtype=np.float64)
def xy_to_lonlat_trajectory(xy_seq, anchor_lon, anchor_lat):
    _, to_lonlat = make_local_frame(anchor_lon, anchor_lat)
    predicted_lons = [anchor_lon]
    predicted_lats = [anchor_lat]
    for x, y in xy_seq:
        lon, lat = to_lonlat(float(x), float(y))
        predicted_lons.append(lon)
        predicted_lats.append(lat)
    return predicted_lons, predicted_lats
def predict_global_mdn_bilstm(model, obs_lons_filtered, obs_lats_filtered, sample_id=0, n_trajectories=N_TRAJECTORIES):
    anchor_lon = obs_lons_filtered[-1]
    anchor_lat = obs_lats_filtered[-1]
    to_xy, _ = make_local_frame(anchor_lon, anchor_lat)
    obs_xy = np.array(
        [
            to_xy(lon, lat)
            for lon, lat in zip(obs_lons_filtered, obs_lats_filtered)
        ],
        dtype=np.float32
    )
    obs_tensor = torch.from_numpy(obs_xy).unsqueeze(0).to(device)
    with torch.no_grad():
        logits_tensor, mu_tensor, sigma_tensor = model(obs_tensor)
    logits = logits_tensor.cpu().numpy()[0]
    mu = mu_tensor.cpu().numpy()[0]
    sigma = sigma_tensor.cpu().numpy()[0]
    sampled_xy, log_scores = sample_mdn_trajectories(
        logits,
        mu,
        sigma,
        n_trajectories=n_trajectories,
        random_seed=GLOBAL_RANDOM_SEED + int(sample_id)
    )
    all_predicted_coordinates = [
        xy_to_lonlat_trajectory(
            xy_sequence,
            anchor_lon,
            anchor_lat
        )
        for xy_sequence in sampled_xy
    ]
    top_index = int(np.argmax(log_scores))
    top1_coordinates = all_predicted_coordinates[top_index]
    return top1_coordinates, all_predicted_coordinates, log_scores
def net_displacement_km(lons, lats):
    return haversine_km_vec(
        lats[0],
        lons[0],
        lats[-1],
        lons[-1]
    )
def print_and_log_summary(df, training_time_sec, lines_out):
    def write_line(text=''):
        lines_out.append(text)
        print(text)
    def metric_line(subset, column, metric_label, unit):
        values = subset[column]
        write_line(
            f"  {metric_label}: "
            f"Mean={values.mean():.3f} {unit}  "
            f"Median={values.median():.3f} {unit}  "
            f"Standard deviation={values.std():.3f}"
        )
    def block(subset, label):
        write_line(f"\n[{label}] Number of samples: {len(subset)}")
        if len(subset) == 0:
            return
        metric_line(subset, 'ADE_top1_km', 'ADE (top-1)', 'km')
        metric_line(subset, 'FDE_top1_km', 'FDE (top-1)', 'km')
        metric_line(subset, 'minADE_km', 'minADE', 'km')
        metric_line(subset, 'minFDE_km', 'minFDE', 'km')
        metric_line(subset, 'heading_MAE_deg', 'Heading MAE', 'degrees')
        metric_line(subset, 'speed_MAE_kmh', 'Speed MAE', 'km/h')
        metric_line(subset, 'MPQR95_km', 'MPQR@95', 'km')
        write_line(
            f"  Average inference time per sample: "
            f"{subset['infer_time_sec'].mean() * 1000:.3f} ms"
        )
    write_line("=" * 78)
    write_line(
        "Baseline: Global Probabilistic MDN-BiLSTM - Summary Results"
    )
    write_line("=" * 78)
    write_line(
        f"Total one-time training duration: {training_time_sec:.1f} seconds "
        f"(a value of 0 indicates that a cached model was loaded without "
        f"retraining during this run)"
    )
    write_line(f"Number of Monte Carlo candidate trajectories: {N_TRAJECTORIES}")
    block(df, "All Samples")
    block(
        df[df['activity_category'] == 'Underway'],
        "Underway Subset"
    )
    write_line("=" * 78)
def main():
    print("=" * 78)
    print("Baseline: Global Probabilistic MDN-BiLSTM")
    print("=" * 78)
    model, training_time_sec = train_or_load_model()
    samples = build_evaluation_samples(TEST_DIR)
    if MATCH_PROPOSED_METHOD_SAMPLES and os.path.exists(PROPOSED_METRICS_CSV):
        proposed_df = pd.read_csv(
            PROPOSED_METRICS_CSV,
            encoding='utf-8-sig'
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
            f"{PROPOSED_METRICS_CSV} was not found. Evaluating all "
            f"{len(samples)} samples."
        )
    rows = []
    for sample in samples:
        start_time = time.time()
        obs_lons_raw = sample['obs_lons']
        obs_lats_raw = sample['obs_lats']
        true_lons = sample['true_future_lons']
        true_lats = sample['true_future_lats']
        filtered_lons = gaussian_filter1d(
            obs_lons_raw,
            sigma=1.0
        )
        filtered_lats = gaussian_filter1d(
            obs_lats_raw,
            sigma=1.0
        )
        top1_coordinates, all_predicted_coordinates, log_scores = (
            predict_global_mdn_bilstm(
                model,
                filtered_lons,
                filtered_lats,
                sample_id=sample['sample_id'],
                n_trajectories=N_TRAJECTORIES
            )
        )
        predicted_lons, predicted_lats = top1_coordinates
        top1_ade, top1_fde, _ = compute_displacement_errors(
            predicted_lons,
            predicted_lats,
            true_lons,
            true_lats
        )
        ensemble_metrics = compute_ensemble_metrics(
            all_predicted_coordinates,
            true_lons,
            true_lats
        )
        mpqr95, _ = compute_mpqr(
            all_predicted_coordinates,
            true_lons,
            true_lats,
            confidence=0.95
        )
        heading_mae, speed_mae = compute_heading_speed_error(
            predicted_lons,
            predicted_lats,
            true_lons,
            true_lats
        )
        classification = classify_sample(
            true_lons,
            true_lats,
            filtered_lons[-1],
            filtered_lats[-1]
        )
        net_displacement = net_displacement_km(
            true_lons,
            true_lats
        )
        inference_time = time.time() - start_time
        row = {
            'sample_id': sample['sample_id'],
            'source_file': sample['source_file'],
            'mmsi': sample['mmsi'],
            'window_start': sample['window_start'],
            'ADE_top1_km': top1_ade,
            'FDE_top1_km': top1_fde,
            'minADE_km': ensemble_metrics['min_ade'],
            'minFDE_km': ensemble_metrics['min_fde'],
            'meanADE_km': ensemble_metrics['mean_ade'],
            'meanFDE_km': ensemble_metrics['mean_fde'],
            'MPQR95_km': mpqr95,
            'heading_MAE_deg': heading_mae,
            'speed_MAE_kmh': speed_mae,
            'ADE_km': top1_ade,
            'FDE_km': top1_fde,
            'net_displacement_km': net_displacement,
            'activity_category': (
                'Underway'
                if net_displacement >= ACTIVITY_THRESHOLD_KM
                else 'Near-stationary'
            ),
            'turn_category': classification['turn_category'],
            'speed_category': classification['speed_category'],
            'train_time_sec': 0.0,
            'global_training_time_sec': training_time_sec,
            'infer_time_sec': inference_time,
            'top1_log_score': float(np.max(log_scores)),
        }
        rows.append(row)
        print(
            f"[Sample {sample['sample_id']}] "
            f"ADE={top1_ade:.3f} km "
            f"FDE={top1_fde:.3f} km "
            f"minADE={ensemble_metrics['min_ade']:.3f} km "
            f"MPQR95={mpqr95:.3f} km "
            f"Inference time={inference_time * 1000:.1f} ms"
        )
    df = pd.DataFrame(rows)
    df.to_csv(
        RESULT_CSV,
        index=False,
        encoding='utf-8-sig'
    )
    print(f"\n[Saved] {RESULT_CSV} ({len(df)} records)")
    lines = []
    print_and_log_summary(
        df,
        training_time_sec,
        lines
    )
    with open(SUMMARY_TXT, 'w', encoding='utf-8') as file:
        file.write('\n'.join(lines))
    print(f"\n[Saved] {SUMMARY_TXT}")
if __name__ == '__main__':
    main()
