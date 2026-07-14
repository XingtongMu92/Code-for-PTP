import os
import time
from math import radians, cos
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from main import (
    build_evaluation_samples, TEST_DIR, OUTPUT_ROOT,
    PREDICTION_STEPS, haversine_km_vec,
    compute_displacement_errors, compute_heading_speed_error, classify_sample,
)
RESULT_CSV = os.path.join(OUTPUT_ROOT, "baseline_cv_kf_metrics.csv")
SUMMARY_TXT = os.path.join(OUTPUT_ROOT, "baseline_cv_kf_summary.txt")
ACTIVITY_THRESHOLD_KM = 1.0
MATCH_PROPOSED_METHOD_SAMPLES = True
PROPOSED_METRICS_CSV = os.path.join(OUTPUT_ROOT, "per_sample_metrics.csv")
Q_SCALE = 5e-4
MEAS_NOISE_STD_KM = 0.05
INIT_VEL_VARIANCE = 1.0
def filter_observation(obs_lons, obs_lats, sigma=1.0):
    return gaussian_filter1d(obs_lons, sigma=sigma), gaussian_filter1d(obs_lats, sigma=sigma)
def kf_predict(x, P, F, Q):
    x = F @ x
    P = F @ P @ F.T + Q
    return x, P
def kf_update(x, P, z, H, R):
    y = z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ y
    P = (np.eye(len(x)) - K @ H) @ P
    return x, P
def run_cv_kf(obs_lons, obs_lats, n_future_steps, q_scale=Q_SCALE, meas_noise_km=MEAS_NOISE_STD_KM):
    anchor_lon, anchor_lat = obs_lons[-1], obs_lats[-1]
    coslat0 = cos(radians(anchor_lat))
    def to_xy(lon, lat):
        return (lon - anchor_lon) * 111.0 * coslat0, (lat - anchor_lat) * 111.0
    def to_lonlat(x, y):
        return anchor_lon + x / (111.0 * coslat0), anchor_lat + y / 111.0
    xy = np.array([to_xy(lo, la) for lo, la in zip(obs_lons, obs_lats)], dtype=float)
    dt = 1.0
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=float)
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]], dtype=float)
    Q = q_scale * np.array([[dt ** 4 / 4, 0, dt ** 3 / 2, 0],
                            [0, dt ** 4 / 4, 0, dt ** 3 / 2],
                            [dt ** 3 / 2, 0, dt ** 2, 0],
                            [0, dt ** 3 / 2, 0, dt ** 2]], dtype=float)
    R = (meas_noise_km ** 2) * np.eye(2)
    x0, y0 = xy[0]
    if len(xy) >= 2:
        vx0, vy0 = xy[1] - xy[0]
    else:
        vx0, vy0 = 0.0, 0.0
    state = np.array([x0, y0, vx0, vy0], dtype=float)
    P = np.diag([meas_noise_km ** 2, meas_noise_km ** 2, INIT_VEL_VARIANCE, INIT_VEL_VARIANCE])
    for t in range(1, len(xy)):
        state, P = kf_predict(state, P, F, Q)
        state, P = kf_update(state, P, xy[t], H, R)
    pred_lons, pred_lats = [anchor_lon], [anchor_lat]
    pos_cov_list = []
    for _ in range(n_future_steps):
        state, P = kf_predict(state, P, F, Q)
        lon, lat = to_lonlat(state[0], state[1])
        pred_lons.append(lon)
        pred_lats.append(lat)
        pos_cov_list.append(P[:2, :2].copy())
    return pred_lons, pred_lats, pos_cov_list
def analytic_mpqr(pos_cov_list, confidence=0.95):
    radii = []
    for cov in pos_cov_list:
        sigma_eff2 = 0.5 * np.trace(cov)
        r = float(np.sqrt(max(-2.0 * sigma_eff2 * np.log(1 - confidence), 0.0)))
        radii.append(r)
    return float(np.mean(radii))
def net_displacement_km(lons, lats):
    return haversine_km_vec(lats[0], lons[0], lats[-1], lons[-1])
def print_and_log_summary(df, lines_out):
    def W(s=''):
        lines_out.append(s)
        print(s)
    def block(sub, label):
        W(f"\n[{label}] Number of samples: {len(sub)}")
        for col, metric_label, unit in [('ADE_km', 'ADE', 'km'),
                                        ('FDE_km', 'FDE', 'km'),
                                        ('heading_MAE_deg', 'Heading MAE', 'degrees'),
                                        ('speed_MAE_kmh', 'Speed MAE', 'km/h'),
                                        ('MPQR95_km', 'MPQR@95 (Analytical Approximation)', 'km')]:
            v = sub[col]
            W(f"  {metric_label}: Mean={v.mean():.3f} {unit}  Median={v.median():.3f} {unit}  Standard deviation={v.std():.3f}")
        W(f"  Average inference time per sample: {sub['infer_time_sec'].mean() * 1000:.3f} ms")
    W("=" * 78)
    W("Baseline: Constant-Velocity Kalman Filter (CV-KF) - Summary Results")
    W("=" * 78)
    W(f"Kalman filter hyperparameters: Q_SCALE={Q_SCALE}, MEAS_NOISE_STD_KM={MEAS_NOISE_STD_KM}, INIT_VEL_VARIANCE={INIT_VEL_VARIANCE} (fixed default values without per-sample tuning)")
    block(df, "All Samples")
    block(df[df['activity_category'] == 'Underway'], "Underway Subset")
    W("=" * 78)
def main():
    print("=" * 78)
    print("Baseline: Constant-Velocity Kalman Filter (CV-KF)")
    print("=" * 78)
    samples = build_evaluation_samples(TEST_DIR)
    if MATCH_PROPOSED_METHOD_SAMPLES and os.path.exists(PROPOSED_METRICS_CSV):
        proposed_df = pd.read_csv(PROPOSED_METRICS_CSV, encoding='utf-8-sig')
        keep_ids = set(proposed_df['sample_id'].tolist())
        samples = [s for s in samples if s['sample_id'] in keep_ids]
        print(f"[Alignment] Detected {PROPOSED_METRICS_CSV}. Evaluating only the {len(samples)} samples listed in that file to ensure comparison with the proposed model uses exactly the same test instances.")
    else:
        print(f"[Notice] Sample alignment is disabled or {PROPOSED_METRICS_CSV} was not found. Evaluating all {len(samples)} samples.")
    rows = []
    for sample in samples:
        t0 = time.time()
        obs_lons_raw, obs_lats_raw = sample['obs_lons'], sample['obs_lats']
        true_lons, true_lats = sample['true_future_lons'], sample['true_future_lats']
        filtered_lons, filtered_lats = filter_observation(obs_lons_raw, obs_lats_raw)
        pred_lons, pred_lats, pos_cov_list = run_cv_kf(filtered_lons, filtered_lats, PREDICTION_STEPS)
        ade, fde, _ = compute_displacement_errors(pred_lons, pred_lats, true_lons, true_lats)
        heading_mae, speed_mae = compute_heading_speed_error(pred_lons, pred_lats, true_lons, true_lats)
        mpqr95 = analytic_mpqr(pos_cov_list, confidence=0.95)
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
            'MPQR95_km': mpqr95,
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
    print_and_log_summary(df, lines)
    with open(SUMMARY_TXT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\n[Saved] {SUMMARY_TXT}")
if __name__ == '__main__':
    main()



