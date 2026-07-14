import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATE_START = "2025-01-01"
DATE_END = "2025-01-20"
FILE_NAME_TEMPLATE = "ais-{date}.txt"

OUTPUT_ROOT = "ais_dataset"
REFERENCE_DIR = os.path.join(
    OUTPUT_ROOT,
    "reference_trajectories",
)
TEST_DIR = os.path.join(
    OUTPUT_ROOT,
    "test_trajectories",
)
MANIFEST_PATH = os.path.join(
    OUTPUT_ROOT,
    "dataset_manifest.csv",
)
REPORT_PATH = os.path.join(
    OUTPUT_ROOT,
    "dataset_summary_report.txt",
)
FUNNEL_PLOT_PATH = os.path.join(
    OUTPUT_ROOT,
    "dataset_construction_funnel.png",
)
STATS_PLOT_PATH = os.path.join(
    OUTPUT_ROOT,
    "dataset_structure_overview.png",
)

TIME_SPLIT_RATIO = 0.8

VOYAGE_GAP_MINUTES = 30
MAX_SPEED_KNOTS = 50
MIN_RAW_POINTS = 10
MIN_NET_DISPLACEMENT_KM = 2.0
RESAMPLE_FREQ = "5min"
MIN_RESAMPLED_POINTS = 40

PORT_NAME = (
    "Houston-Galveston Ship Channel and Approaches"
)

PORT_BBOX = {
    "lon_min": -95.50,
    "lon_max": -94.45,
    "lat_min": 29.10,
    "lat_max": 29.95,
}

FILTER_VESSEL_TYPE = True
VESSEL_TYPE_RANGE = (70, 89)

CHUNK_SIZE = 500_000
EARTH_RADIUS_KM = 6371.0

COLUMN_ALIASES = {
    "mmsi": [
        "mmsi",
        "MMSI",
    ],
    "time": [
        "base_date_time",
        "basedatetime",
        "BaseDateTime",
        "time",
    ],
    "lon": [
        "longitude",
        "lon",
        "LON",
    ],
    "lat": [
        "latitude",
        "lat",
        "LAT",
    ],
    "vessel_type": [
        "vessel_type",
        "vesseltype",
        "VesselType",
    ],
}


def build_file_list():
    dates = pd.date_range(
        DATE_START,
        DATE_END,
        freq="D",
    )

    files = []
    missing = []

    for date in dates:
        file_name = FILE_NAME_TEMPLATE.format(
            date=date.strftime("%Y-%m-%d")
        )

        if os.path.exists(file_name):
            files.append(file_name)
        else:
            missing.append(file_name)

    if missing:
        print(
            f"[Warning] {len(missing)} files were not found "
            f"and will be skipped: {missing}"
        )

    if not files:
        raise FileNotFoundError(
            "No AIS files were found in the selected date range. "
            "Expected file names in the format ais-YYYY-MM-DD.txt."
        )

    print(
        f"[Info] Found {len(files)} data files: {files}"
    )

    return files


def normalize_columns(df):
    lower_columns = {
        column.lower(): column
        for column in df.columns
    }

    rename_map = {}

    for standard_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias.lower() in lower_columns:
                rename_map[
                    lower_columns[alias.lower()]
                ] = standard_name
                break

    df = df.rename(
        columns=rename_map
    )

    required_columns = [
        "mmsi",
        "time",
        "lon",
        "lat",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return df


def filter_by_bbox(df, bbox):
    mask = (
        df["lon"].between(
            bbox["lon_min"],
            bbox["lon_max"],
        )
        & df["lat"].between(
            bbox["lat_min"],
            bbox["lat_max"],
        )
    )

    return df[mask]


def load_and_filter(files):
    all_chunks = []
    total_raw = 0

    for file_path in files:
        print(
            f"[Info] Reading {file_path}..."
        )

        rows_in_file = 0

        reader = pd.read_csv(
            file_path,
            chunksize=CHUNK_SIZE,
            low_memory=False,
        )

        for chunk in reader:
            total_raw += len(chunk)
            rows_in_file += len(chunk)

            chunk = normalize_columns(
                chunk
            )

            chunk = chunk.dropna(
                subset=[
                    "mmsi",
                    "time",
                    "lon",
                    "lat",
                ]
            )

            chunk["time"] = pd.to_datetime(
                chunk["time"],
                errors="coerce",
            )
            chunk["lon"] = pd.to_numeric(
                chunk["lon"],
                errors="coerce",
            )
            chunk["lat"] = pd.to_numeric(
                chunk["lat"],
                errors="coerce",
            )

            chunk = chunk.dropna(
                subset=[
                    "time",
                    "lon",
                    "lat",
                ]
            )

            chunk = chunk[
                chunk["lon"].between(
                    -180,
                    180,
                )
                & chunk["lat"].between(
                    -90,
                    90,
                )
            ]

            chunk = chunk[
                ~(
                    (chunk["lon"].abs() < 1e-6)
                    & (chunk["lat"].abs() < 1e-6)
                )
            ]

            if "vessel_type" in chunk.columns:
                chunk["vessel_type"] = pd.to_numeric(
                    chunk["vessel_type"],
                    errors="coerce",
                )

            chunk = filter_by_bbox(
                chunk,
                PORT_BBOX,
            )

            if len(chunk) > 0:
                if "vessel_type" in chunk.columns:
                    selected_columns = [
                        "mmsi",
                        "time",
                        "lon",
                        "lat",
                        "vessel_type",
                    ]
                else:
                    selected_columns = [
                        "mmsi",
                        "time",
                        "lon",
                        "lat",
                    ]

                all_chunks.append(
                    chunk[selected_columns]
                )

        print(
            f"    Raw rows in file: {rows_in_file:,}"
        )

    if not all_chunks:
        raise ValueError(
            "No records remained after cleaning and "
            "geographic filtering."
        )

    df = pd.concat(
        all_chunks,
        ignore_index=True,
    )

    df = df.drop_duplicates(
        subset=[
            "mmsi",
            "time",
            "lon",
            "lat",
        ]
    )

    print(
        f"[Info] Total raw records across {len(files)} files: "
        f"{total_raw:,}; valid records inside the selected area: "
        f"{len(df):,}"
    )

    return df, total_raw


def haversine_np(
    lat1,
    lon1,
    lat2,
    lon2,
):
    lat1, lon1, lat2, lon2 = map(
        np.radians,
        [
            lat1,
            lon1,
            lat2,
            lon2,
        ],
    )

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(delta_lon / 2) ** 2
    )

    c = 2 * np.arcsin(
        np.sqrt(
            np.clip(
                a,
                0,
                1,
            )
        )
    )

    return EARTH_RADIUS_KM * c


def segment_voyages(df):
    df = df.sort_values(
        [
            "mmsi",
            "time",
        ]
    ).reset_index(
        drop=True
    )

    previous_lon = df.groupby(
        "mmsi"
    )["lon"].shift(1)

    previous_lat = df.groupby(
        "mmsi"
    )["lat"].shift(1)

    previous_time = df.groupby(
        "mmsi"
    )["time"].shift(1)

    time_difference_minutes = (
        df["time"] - previous_time
    ).dt.total_seconds() / 60.0

    time_difference_hours = (
        time_difference_minutes / 60.0
    )

    distance_km = haversine_np(
        previous_lat.values,
        previous_lon.values,
        df["lat"].values,
        df["lon"].values,
    )

    with np.errstate(
        invalid="ignore",
        divide="ignore",
    ):
        implied_speed_kmh = np.where(
            time_difference_hours.values > 0,
            distance_km
            / time_difference_hours.values,
            np.nan,
        )

    implied_speed_knots = (
        implied_speed_kmh / 1.852
    )

    new_voyage_flag = (
        time_difference_minutes.isna()
        | (
            time_difference_minutes
            > VOYAGE_GAP_MINUTES
        )
        | (
            implied_speed_knots
            > MAX_SPEED_KNOTS
        )
    )

    voyage_sequence = (
        new_voyage_flag
        .groupby(df["mmsi"])
        .cumsum()
    )

    df["voyage_id"] = (
        df["mmsi"].astype(str)
        + "_"
        + voyage_sequence.astype(str)
    )

    return df


def get_voyage_vessel_type(df):
    def mode_or_nan(series):
        series = series.dropna()

        if len(series) == 0:
            return np.nan

        return series.mode().iloc[0]

    if "vessel_type" not in df.columns:
        return pd.Series(
            dtype=float
        )

    return df.groupby(
        "voyage_id"
    )["vessel_type"].apply(
        mode_or_nan
    )


def net_displacement_km(voyage_df_sorted):
    start_lon = voyage_df_sorted[
        "lon"
    ].iloc[0]
    start_lat = voyage_df_sorted[
        "lat"
    ].iloc[0]

    end_lon = voyage_df_sorted[
        "lon"
    ].iloc[-1]
    end_lat = voyage_df_sorted[
        "lat"
    ].iloc[-1]

    return haversine_np(
        np.array([start_lat]),
        np.array([start_lon]),
        np.array([end_lat]),
        np.array([end_lon]),
    )[0]


def resample_voyage_df(
    voyage_df,
    freq=RESAMPLE_FREQ,
):
    voyage = (
        voyage_df
        .sort_values("time")
        .set_index("time")
    )

    voyage = voyage[
        ~voyage.index.duplicated(
            keep="first"
        )
    ]

    if len(voyage) < 2:
        return None

    return (
        voyage[
            [
                "lon",
                "lat",
            ]
        ]
        .resample(freq)
        .mean()
        .interpolate(
            method="linear"
        )
        .dropna()
    )


def generate_report(
    total_raw,
    n_voyages_raw,
    n_voyages_vt,
    n_after_raw_filter,
    n_after_disp_filter,
    n_final,
    ref_count,
    test_count,
    manifest_df,
    global_min_t,
    global_max_t,
    split_time,
):
    lines = []

    def write(text=""):
        lines.append(text)
        print(text)

    n_days = (
        pd.Timestamp(DATE_END)
        - pd.Timestamp(DATE_START)
    ).days + 1

    write("=" * 70)
    write(
        "AIS Trajectory Dataset Construction Report"
    )
    write("=" * 70)
    write(
        f"Generated at: "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    write(
        f"Data period: {DATE_START} to {DATE_END} "
        f"({n_days} days)"
    )
    write(
        f"Geographic area: {PORT_NAME}"
    )
    write(
        f"  Longitude range: "
        f"[{PORT_BBOX['lon_min']}, {PORT_BBOX['lon_max']}]"
    )
    write(
        f"  Latitude range: "
        f"[{PORT_BBOX['lat_min']}, {PORT_BBOX['lat_max']}]"
    )
    write(
        f"Target vessel types: "
        f"VesselType {VESSEL_TYPE_RANGE[0]}-"
        f"{VESSEL_TYPE_RANGE[1]}"
    )
    write("")
    write("-" * 70)
    write("Dataset Construction Funnel")
    write("-" * 70)
    write(
        f"1. Total raw records: {total_raw:,}"
    )
    write(
        f"2. Raw voyages inside the selected area: "
        f"{n_voyages_raw:,}"
    )
    write(
        f"3. Cargo and tanker voyages: "
        f"{n_voyages_vt:,} "
        f"({n_voyages_vt / max(n_voyages_raw, 1) * 100:.1f}%)"
    )
    write(
        f"4. Voyages with at least {MIN_RAW_POINTS} raw points: "
        f"{n_after_raw_filter:,} "
        f"({n_after_raw_filter / max(n_voyages_vt, 1) * 100:.1f}%)"
    )
    write(
        f"5. Voyages with at least "
        f"{MIN_NET_DISPLACEMENT_KM} km net displacement: "
        f"{n_after_disp_filter:,} "
        f"({n_after_disp_filter / max(n_after_raw_filter, 1) * 100:.1f}%)"
    )
    write(
        f"6. Final voyages with at least "
        f"{MIN_RESAMPLED_POINTS} resampled points: "
        f"{n_final:,} "
        f"({n_final / max(n_after_disp_filter, 1) * 100:.1f}%)"
    )
    write("")
    write("-" * 70)
    write("Reference and Test Split")
    write("-" * 70)
    write(
        f"Split method: chronological voyage start time, "
        f"first {int(TIME_SPLIT_RATIO * 100)}% to reference "
        f"and final {int((1 - TIME_SPLIT_RATIO) * 100)}% to test"
    )
    write(
        f"Global time range: "
        f"{global_min_t} to {global_max_t}"
    )
    write(
        f"Split timestamp: {split_time}"
    )
    write(
        f"Reference voyages: {ref_count:,} "
        f"({ref_count / n_final * 100:.1f}%)"
    )
    write(
        f"Test voyages: {test_count:,} "
        f"({test_count / n_final * 100:.1f}%)"
    )
    write("")

    for label in [
        "reference",
        "test",
    ]:
        subset = manifest_df[
            manifest_df["split"] == label
        ]

        write("-" * 70)
        write(
            f"{label.capitalize()} Subset Statistics"
        )
        write("-" * 70)

        if len(subset) == 0:
            write("No data")
            write("")
            continue

        write(
            f"Voyage count: {len(subset):,}"
        )
        write(
            f"Unique MMSI count: "
            f"{subset['mmsi'].nunique():,}"
        )
        write(
            f"Time range: "
            f"{subset['start_time'].min()} to "
            f"{subset['end_time'].max()}"
        )
        write(
            "Resampled points per voyage: "
            f"mean={subset['n_points_resampled'].mean():.1f}, "
            f"median={subset['n_points_resampled'].median():.1f}, "
            f"min={subset['n_points_resampled'].min():.0f}, "
            f"max={subset['n_points_resampled'].max():.0f}"
        )

        duration_minutes = (
            pd.to_datetime(
                subset["end_time"]
            )
            - pd.to_datetime(
                subset["start_time"]
            )
        ).dt.total_seconds() / 60.0

        write(
            "Voyage duration in minutes: "
            f"mean={duration_minutes.mean():.1f}, "
            f"median={duration_minutes.median():.1f}, "
            f"min={duration_minutes.min():.1f}, "
            f"max={duration_minutes.max():.1f}"
        )
        write("")

    reference_mmsi = set(
        manifest_df[
            manifest_df["split"] == "reference"
        ]["mmsi"]
    )

    test_mmsi = set(
        manifest_df[
            manifest_df["split"] == "test"
        ]["mmsi"]
    )

    overlap = (
        reference_mmsi & test_mmsi
    )

    write("-" * 70)
    write(
        "MMSI Overlap Between Reference and Test Sets"
    )
    write("-" * 70)
    write(
        f"Unique reference vessels: "
        f"{len(reference_mmsi)}"
    )
    write(
        f"Unique test vessels: "
        f"{len(test_mmsi)}"
    )
    write(
        f"Overlapping vessels: "
        f"{len(overlap)} "
        f"({len(overlap) / max(len(test_mmsi), 1) * 100:.1f}% "
        f"of test vessels)"
    )
    write(
        "Overlap is expected because the same vessel may appear "
        "in different voyages during the reference and test periods."
    )
    write(
        "Reference voyages occur earlier than test voyages, "
        "so this does not introduce future-data leakage."
    )
    write("")
    write("-" * 70)
    write("Output Structure")
    write("-" * 70)
    write(
        f"{OUTPUT_ROOT}/"
    )
    write(
        f"  |-- reference_trajectories/ "
        f"({ref_count} files)"
    )
    write(
        f"  |-- test_trajectories/ "
        f"({test_count} files)"
    )
    write(
        "  |-- dataset_manifest.csv"
    )
    write(
        "  |-- dataset_summary_report.txt"
    )
    write(
        "  |-- dataset_construction_funnel.png"
    )
    write(
        "  `-- dataset_structure_overview.png"
    )
    write("")
    write(
        "Each trajectory CSV contains LONGITUDE, LATITUDE, "
        "and TIME columns."
    )
    write(
        f"Sampling interval: {RESAMPLE_FREQ}"
    )
    write("=" * 70)

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "\n".join(lines)
        )

    print(
        f"\n[Saved] {REPORT_PATH}"
    )


def plot_funnel(
    n_voyages_raw,
    n_voyages_vt,
    n_after_raw_filter,
    n_after_disp_filter,
    n_final,
):
    stages = [
        "Raw Voyages",
        "Cargo and Tanker",
        "Raw Point Filter",
        "Displacement Filter",
        "Final Resampled Voyages",
    ]

    values = [
        n_voyages_raw,
        n_voyages_vt,
        n_after_raw_filter,
        n_after_disp_filter,
        n_final,
    ]

    figure, axis = plt.subplots(
        figsize=(11, 6)
    )

    y_positions = np.arange(
        len(stages)
    )

    bars = axis.barh(
        y_positions,
        values,
        color=plt.cm.viridis(
            np.linspace(
                0.2,
                0.85,
                len(stages),
            )
        ),
    )

    axis.set_yticks(
        y_positions
    )
    axis.set_yticklabels(
        stages
    )
    axis.invert_yaxis()

    for bar, value in zip(
        bars,
        values,
    ):
        axis.text(
            bar.get_width(),
            bar.get_y()
            + bar.get_height() / 2,
            f"  {value:,}",
            va="center",
            fontsize=11,
        )

    axis.set_xlabel(
        "Number of Voyages"
    )
    axis.set_title(
        f"Dataset Construction Funnel\n"
        f"({PORT_NAME}, VesselType "
        f"{VESSEL_TYPE_RANGE[0]}-{VESSEL_TYPE_RANGE[1]}, "
        f"{DATE_START} to {DATE_END})",
        fontsize=13,
        fontweight="bold",
    )
    axis.grid(
        True,
        axis="x",
        alpha=0.3,
    )

    plt.tight_layout()
    plt.savefig(
        FUNNEL_PLOT_PATH,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    print(
        f"[Saved] {FUNNEL_PLOT_PATH}"
    )


def plot_structure_overview(
    manifest_df,
):
    colors_map = {
        "reference": "#378ADD",
        "test": "#D85A30",
    }

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
    )

    axis = axes[0, 0]

    counts = manifest_df[
        "split"
    ].value_counts()

    axis.bar(
        counts.index,
        counts.values,
        color=[
            colors_map.get(
                label,
                "gray",
            )
            for label in counts.index
        ],
    )

    for index, (
        label,
        value,
    ) in enumerate(
        counts.items()
    ):
        axis.text(
            index,
            value,
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    axis.set_title(
        "Reference vs Test Voyage Count"
    )
    axis.set_ylabel(
        "Number of Voyages"
    )
    axis.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    axis = axes[0, 1]

    for label, color in colors_map.items():
        subset = manifest_df[
            manifest_df["split"] == label
        ]["n_points_resampled"]

        if len(subset) > 0:
            axis.hist(
                subset,
                bins=20,
                alpha=0.6,
                label=label,
                color=color,
                edgecolor="black",
            )

    axis.axvline(
        MIN_RESAMPLED_POINTS,
        color="black",
        linestyle="--",
        label=(
            f"Minimum={MIN_RESAMPLED_POINTS}"
        ),
    )

    axis.set_title(
        "Resampled Points per Voyage"
    )
    axis.set_xlabel(
        "Number of Points"
    )
    axis.set_ylabel(
        "Number of Voyages"
    )
    axis.legend()
    axis.grid(
        True,
        alpha=0.3,
    )

    axis = axes[1, 0]

    manifest_df[
        "start_date"
    ] = pd.to_datetime(
        manifest_df["start_time"]
    ).dt.date

    daily_counts = (
        manifest_df
        .groupby(
            [
                "start_date",
                "split",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    daily_counts = daily_counts.reindex(
        columns=[
            "reference",
            "test",
        ],
        fill_value=0,
    )

    daily_counts.plot(
        kind="bar",
        stacked=True,
        ax=axis,
        color=[
            colors_map.get(
                column,
                "gray",
            )
            for column in daily_counts.columns
        ],
    )

    axis.set_title(
        "Daily Voyage Distribution"
    )
    axis.set_xlabel(
        "Date"
    )
    axis.set_ylabel(
        "Number of Voyages"
    )
    axis.tick_params(
        axis="x",
        rotation=45,
    )
    axis.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    axis = axes[1, 1]

    manifest_df[
        "duration_min"
    ] = (
        pd.to_datetime(
            manifest_df["end_time"]
        )
        - pd.to_datetime(
            manifest_df["start_time"]
        )
    ).dt.total_seconds() / 60.0

    for label, color in colors_map.items():
        subset = manifest_df[
            manifest_df["split"] == label
        ]["duration_min"]

        if len(subset) > 0:
            axis.hist(
                subset,
                bins=20,
                alpha=0.6,
                label=label,
                color=color,
                edgecolor="black",
            )

    axis.set_title(
        "Voyage Duration Distribution"
    )
    axis.set_xlabel(
        "Duration in Minutes"
    )
    axis.set_ylabel(
        "Number of Voyages"
    )
    axis.legend()
    axis.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()
    plt.savefig(
        STATS_PLOT_PATH,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    print(
        f"[Saved] {STATS_PLOT_PATH}"
    )


def main():
    print("=" * 70)
    print(
        "AIS Reference and Test Dataset Construction"
    )
    print("=" * 70)

    os.makedirs(
        REFERENCE_DIR,
        exist_ok=True,
    )
    os.makedirs(
        TEST_DIR,
        exist_ok=True,
    )

    files = build_file_list()

    df, total_raw = load_and_filter(
        files
    )

    print(
        "\n[Step] Segmenting voyages..."
    )

    df = segment_voyages(
        df
    )

    n_voyages_raw = df[
        "voyage_id"
    ].nunique()

    print(
        f"Raw voyages inside the selected area: "
        f"{n_voyages_raw:,}"
    )

    print(
        "\n[Step] Filtering vessel types..."
    )

    voyage_vessel_types = (
        get_voyage_vessel_type(df)
    )

    if (
        FILTER_VESSEL_TYPE
        and len(voyage_vessel_types) > 0
    ):
        valid_vessel_type_ids = set(
            voyage_vessel_types[
                voyage_vessel_types.between(
                    *VESSEL_TYPE_RANGE
                )
            ].index
        )
    else:
        valid_vessel_type_ids = set(
            df["voyage_id"].unique()
        )

    vessel_filtered_df = df[
        df["voyage_id"].isin(
            valid_vessel_type_ids
        )
    ].copy()

    n_voyages_vt = (
        vessel_filtered_df[
            "voyage_id"
        ].nunique()
    )

    print(
        f"Cargo and tanker voyages: "
        f"{n_voyages_vt:,}"
    )

    print(
        f"\n[Step] Applying minimum raw point filter "
        f"({MIN_RAW_POINTS})..."
    )

    point_counts = (
        vessel_filtered_df
        .groupby("voyage_id")
        .size()
    )

    raw_point_ids = set(
        point_counts[
            point_counts
            >= MIN_RAW_POINTS
        ].index
    )

    vessel_filtered_df = (
        vessel_filtered_df[
            vessel_filtered_df[
                "voyage_id"
            ].isin(raw_point_ids)
        ]
    )

    n_after_raw_filter = (
        vessel_filtered_df[
            "voyage_id"
        ].nunique()
    )

    print(
        f"Voyages passing the raw point filter: "
        f"{n_after_raw_filter:,}"
    )

    print(
        f"\n[Step] Applying minimum displacement filter "
        f"({MIN_NET_DISPLACEMENT_KM} km)..."
    )

    voyage_groups = {
        voyage_id: segment.sort_values(
            "time"
        )
        for voyage_id, segment
        in vessel_filtered_df.groupby(
            "voyage_id"
        )
    }

    displacement_ids = [
        voyage_id
        for voyage_id, segment
        in voyage_groups.items()
        if net_displacement_km(segment)
        >= MIN_NET_DISPLACEMENT_KM
    ]

    n_after_disp_filter = len(
        displacement_ids
    )

    print(
        f"Voyages passing the displacement filter: "
        f"{n_after_disp_filter:,}"
    )

    print(
        f"\n[Step] Resampling at {RESAMPLE_FREQ} "
        f"and applying minimum point filter "
        f"({MIN_RESAMPLED_POINTS})..."
    )

    final_records = []

    for voyage_id in displacement_ids:
        segment = voyage_groups[
            voyage_id
        ]

        resampled = resample_voyage_df(
            segment
        )

        if (
            resampled is None
            or len(resampled)
            < MIN_RESAMPLED_POINTS
        ):
            continue

        vessel_type = (
            voyage_vessel_types.get(
                voyage_id,
                np.nan,
            )
        )

        final_records.append(
            {
                "voyage_id": voyage_id,
                "mmsi": segment[
                    "mmsi"
                ].iloc[0],
                "vessel_type": vessel_type,
                "start_time": resampled.index[0],
                "end_time": resampled.index[-1],
                "n_points_resampled": len(
                    resampled
                ),
                "resampled_df": resampled,
            }
        )

    n_final = len(
        final_records
    )

    print(
        f"Final voyage count: {n_final:,}"
    )

    if n_final == 0:
        raise ValueError(
            "No voyages passed all filters."
        )

    print(
        f"\n[Step] Splitting reference and test data "
        f"at {int(TIME_SPLIT_RATIO * 100)}:"
        f"{int((1 - TIME_SPLIT_RATIO) * 100)}..."
    )

    all_start_times = [
        record["start_time"]
        for record in final_records
    ]

    global_min_t = min(
        all_start_times
    )
    global_max_t = max(
        all_start_times
    )

    split_time = (
        global_min_t
        + (
            global_max_t
            - global_min_t
        )
        * TIME_SPLIT_RATIO
    )

    print(
        f"Global time range: "
        f"{global_min_t} to {global_max_t}"
    )
    print(
        f"Split timestamp: {split_time}"
    )

    manifest_rows = []
    reference_count = 0
    test_count = 0

    for record in final_records:
        split_label = (
            "reference"
            if record["start_time"]
            < split_time
            else "test"
        )

        output_directory = (
            REFERENCE_DIR
            if split_label == "reference"
            else TEST_DIR
        )

        date_tag = record[
            "start_time"
        ].strftime(
            "%Y%m%d"
        )

        voyage_sequence = record[
            "voyage_id"
        ].split(
            "_"
        )[-1]

        file_name = (
            f"{date_tag}_"
            f"{record['mmsi']}_"
            f"{voyage_sequence}_"
            f"merged_resampled_5min.csv"
        )

        output_path = os.path.join(
            output_directory,
            file_name,
        )

        output_df = (
            record["resampled_df"]
            .reset_index()
            .rename(
                columns={
                    "time": "TIME",
                    "lon": "LONGITUDE",
                    "lat": "LATITUDE",
                }
            )
        )

        output_df["TIME"] = (
            output_df["TIME"]
            .dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        output_df = output_df[
            [
                "LONGITUDE",
                "LATITUDE",
                "TIME",
            ]
        ]

        output_df.to_csv(
            output_path,
            index=False,
        )

        reference_count += (
            split_label == "reference"
        )
        test_count += (
            split_label == "test"
        )

        manifest_rows.append(
            {
                "file_name": file_name,
                "split": split_label,
                "voyage_id": record[
                    "voyage_id"
                ],
                "mmsi": record["mmsi"],
                "vessel_type": record[
                    "vessel_type"
                ],
                "start_time": record[
                    "start_time"
                ],
                "end_time": record[
                    "end_time"
                ],
                "n_points_resampled": record[
                    "n_points_resampled"
                ],
            }
        )

    manifest_df = pd.DataFrame(
        manifest_rows
    )

    manifest_df.to_csv(
        MANIFEST_PATH,
        index=False,
    )

    print(
        f"Reference voyages: "
        f"{reference_count:,} "
        f"({reference_count / n_final * 100:.1f}%)"
    )
    print(
        f"Test voyages: "
        f"{test_count:,} "
        f"({test_count / n_final * 100:.1f}%)"
    )

    generate_report(
        total_raw,
        n_voyages_raw,
        n_voyages_vt,
        n_after_raw_filter,
        n_after_disp_filter,
        n_final,
        reference_count,
        test_count,
        manifest_df,
        global_min_t,
        global_max_t,
        split_time,
    )

    plot_funnel(
        n_voyages_raw,
        n_voyages_vt,
        n_after_raw_filter,
        n_after_disp_filter,
        n_final,
    )

    plot_structure_overview(
        manifest_df
    )

    print(
        "\nDataset construction completed."
    )


if __name__ == "__main__":
    main()
