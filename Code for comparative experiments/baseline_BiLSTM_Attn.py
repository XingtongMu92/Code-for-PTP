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
RESULT_CSV = os.path.join(OUTPUT_ROOT, "baseline_global_bilstm_attn_metrics.csv")
SUMMARY_TXT = os.path.join(OUTPUT_ROOT, "baseline_global_bilstm_attn_summary.txt")
MODEL_CACHE_PATH = os.path.join(OUTPUT_ROOT, "global_bilstm_attn_checkpoint.pt")
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
    for file_path in files:
        df = pd.read_csv(file_path)
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
        raise ValueError(
            f"No training sample pairs could be generated from {len(files)} reference trajectories."
        )
    print(
        f"[Training Data] Generated {len(obs_list)} training sample pairs "
        f"from {len(files)} reference trajectories using sliding windows "
        f"(observation steps: {obs_len}, prediction steps: {pred_len}, stride: {stride})"
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
class BiEncoder(nn.Module):
    def __init__(self, input_size=2, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.fc_h = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_c = nn.Linear(hidden_size * 2, hidden_size)
    def forward(self, x):
        outputs, (hidden_state, cell_state) = self.lstm(x)
        hidden_cat = torch.cat([hidden_state[0], hidden_state[1]], dim=1)
        cell_cat = torch.cat([cell_state[0], cell_state[1]], dim=1)
        initial_hidden = torch.tanh(self.fc_h(hidden_cat))
        initial_cell = torch.tanh(self.fc_c(cell_cat))
        return outputs, initial_hidden, initial_cell
class LuongAttention(nn.Module):
    def __init__(self, hidden_size, encoder_output_size):
        super().__init__()
        self.attn = nn.Linear(encoder_output_size, hidden_size)
    def forward(self, decoder_hidden, encoder_outputs):
        projected_outputs = self.attn(encoder_outputs)
        scores = torch.bmm(
            projected_outputs,
            decoder_hidden.unsqueeze(2),
        ).squeeze(2)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(
            weights.unsqueeze(1),
            encoder_outputs,
        ).squeeze(1)
        return context, weights
class AttnDecoder(nn.Module):
    def __init__(
        self,
        input_size=2,
        hidden_size=HIDDEN_SIZE,
        output_size=2,
        encoder_output_size=None,
    ):
        super().__init__()
        encoder_output_size = encoder_output_size or hidden_size * 2
        self.attention = LuongAttention(hidden_size, encoder_output_size)
        self.cell = nn.LSTMCell(
            input_size + encoder_output_size,
            hidden_size,
        )
        self.fc_out = nn.Linear(
            hidden_size + encoder_output_size,
            output_size,
        )
    def forward(
        self,
        h0,
        c0,
        encoder_outputs,
        n_steps,
        target_seq=None,
        teacher_forcing=True,
    ):
        batch_size = h0.size(0)
        hidden_state = h0
        cell_state = c0
        previous_position = torch.zeros(
            batch_size,
            2,
            device=h0.device,
        )
        outputs = []
        for step in range(n_steps):
            context, _ = self.attention(hidden_state, encoder_outputs)
            cell_input = torch.cat([previous_position, context], dim=1)
            hidden_state, cell_state = self.cell(
                cell_input,
                (hidden_state, cell_state),
            )
            combined = torch.cat([hidden_state, context], dim=1)
            prediction = self.fc_out(combined)
            outputs.append(prediction)
            if teacher_forcing and target_seq is not None:
                previous_position = target_seq[:, step, :]
            else:
                previous_position = prediction
        return torch.stack(outputs, dim=1)
class GlobalBiLSTMAttnModel(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.encoder = BiEncoder(hidden_size=hidden_size)
        self.decoder = AttnDecoder(
            hidden_size=hidden_size,
            encoder_output_size=hidden_size * 2,
        )
    def forward(
        self,
        obs_seq,
        n_steps=PREDICTION_STEPS,
        target_seq=None,
        teacher_forcing=True,
    ):
        encoder_outputs, initial_hidden, initial_cell = self.encoder(obs_seq)
        return self.decoder(
            initial_hidden,
            initial_cell,
            encoder_outputs,
            n_steps,
            target_seq=target_seq,
            teacher_forcing=teacher_forcing,
        )
def train_or_load_model():
    if os.path.exists(MODEL_CACHE_PATH) and not FORCE_RETRAIN:
        print(
            f"[Model] A cached model was detected at {MODEL_CACHE_PATH}. "
            f"Loading it without retraining."
        )
        model = GlobalBiLSTMAttnModel().to(device)
        model.load_state_dict(
            torch.load(MODEL_CACHE_PATH, map_location=device)
        )
        model.eval()
        return model, 0.0
    print("[Model] Starting one-time global model training...")
    obs_arr, fut_arr = build_training_pairs_from_reference(REFERENCE_DIR)
    dataset = GlobalTrajDataset(obs_arr, fut_arr)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    model = GlobalBiLSTMAttnModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    training_start_time = time.time()
    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        number_of_batches = 0
        for obs_batch, fut_batch in loader:
            obs_batch = obs_batch.to(device)
            fut_batch = fut_batch.to(device)
            optimizer.zero_grad()
            predictions = model(
                obs_batch,
                n_steps=fut_batch.size(1),
                target_seq=fut_batch,
                teacher_forcing=True,
            )
            loss = criterion(predictions, fut_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            number_of_batches += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            average_loss = total_loss / max(number_of_batches, 1)
            print(
                f"  Epoch [{epoch + 1}/{NUM_EPOCHS}] "
                f"Training loss (MSE in local kilometer coordinates): "
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
def predict_global_bilstm_attn(model, obs_lons_filtered, obs_lats_filtered):
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
    with torch.no_grad():
        predicted_xy = model(
            obs_tensor,
            n_steps=PREDICTION_STEPS,
            target_seq=None,
            teacher_forcing=False,
        ).cpu().numpy()[0]
    pred_lons = [anchor_lon]
    pred_lats = [anchor_lat]
    for x, y in predicted_xy:
        lon, lat = to_lonlat(float(x), float(y))
        pred_lons.append(lon)
        pred_lats.append(lat)
    return pred_lons, pred_lats
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
            ('ADE_km', 'ADE', 'km'),
            ('FDE_km', 'FDE', 'km'),
            ('heading_MAE_deg', 'Heading MAE', 'degrees'),
            ('speed_MAE_kmh', 'Speed MAE', 'km/h'),
        ]
        for column, metric_label, unit in metrics:
            values = subset[column]
            write_line(
                f"  {metric_label}: "
                f"Mean={values.mean():.3f} {unit}  "
                f"Median={values.median():.3f} {unit}  "
                f"Standard deviation={values.std():.3f}"
            )
        write_line(
            f"  Average inference time per sample: "
            f"{subset['infer_time_sec'].mean() * 1000:.3f} ms"
        )
    write_line("=" * 78)
    write_line(
        "Baseline: Global Bidirectional LSTM with Attention "
        "- Summary Results"
    )
    write_line("=" * 78)
    write_line(
        f"Total one-time training duration: {training_time_sec:.1f} seconds "
        f"(a value of 0 indicates that a cached model was loaded without "
        f"retraining during this run)"
    )
    write_block(df, "All Samples")
    write_block(
        df[df['activity_category'] == 'Underway'],
        "Underway Subset",
    )
    write_line("=" * 78)
def main():
    print("=" * 78)
    print("Baseline: Global Bidirectional LSTM with Attention")
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
            f"[Alignment] Detected {PROPOSED_METRICS_CSV}. "
            f"Evaluating only the {len(samples)} samples listed in that file "
            f"to ensure that the comparison with the proposed model uses "
            f"exactly the same test instances."
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
        pred_lons, pred_lats = predict_global_bilstm_attn(
            model,
            filtered_lons,
            filtered_lats,
        )
        ade, fde, _ = compute_displacement_errors(
            pred_lons,
            pred_lats,
            true_lons,
            true_lats,
        )
        heading_mae, speed_mae = compute_heading_speed_error(
            pred_lons,
            pred_lats,
            true_lons,
            true_lats,
        )
        classification_info = classify_sample(
            true_lons,
            true_lats,
            filtered_lons[-1],
            filtered_lats[-1],
        )
        net_displacement = net_displacement_km(
            true_lons,
            true_lats,
        )
        rows.append({
            'sample_id': sample['sample_id'],
            'source_file': sample['source_file'],
            'mmsi': sample['mmsi'],
            'ADE_km': ade,
            'FDE_km': fde,
            'heading_MAE_deg': heading_mae,
            'speed_MAE_kmh': speed_mae,
            'net_displacement_km': net_displacement,
            'activity_category': (
                'Underway'
                if net_displacement >= ACTIVITY_THRESHOLD_KM
                else 'Near-stationary'
            ),
            'turn_category': classification_info['turn_category'],
            'speed_category': classification_info['speed_category'],
            'infer_time_sec': time.time() - inference_start_time,
        })
    df = pd.DataFrame(rows)
    df.to_csv(
        RESULT_CSV,
        index=False,
        encoding='utf-8-sig',
    )
    print(f"\n[Saved] {RESULT_CSV} ({len(df)} records)")
    lines = []
    print_and_log_summary(df, training_time_sec, lines)
    with open(SUMMARY_TXT, 'w', encoding='utf-8') as file:
        file.write('\n'.join(lines))
    print(f"\n[Saved] {SUMMARY_TXT}")
if __name__ == '__main__':
    main()
