from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


# =========================================================
# =========================================================
INPUT_FILE = Path("data/raw/xingguang/xingguang_daily.csv")

OUTPUT_DIR = Path("data/processed/xingguang")
OUTPUT_NPZ = OUTPUT_DIR / "xingguang_image_dataset.npz"
OUTPUT_META = OUTPUT_DIR / "xingguang_image_metadata.csv"

WINDOW_SIZE = 60

FEATURE_COLUMNS = [
    "close",
    "ma5",
    "ma20",
    "ma60",
    "volume",
    "ret_1d",
]

LABEL_5D_COL = "label_5d"
LABEL_10D_COL = "label_10d"
LABEL_20D_COL = "label_20d"


# =========================================================
# =========================================================
def load_daily_data(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f": {file_path}")

    df = pd.read_csv(file_path)
    if "trade_date" not in df.columns:
        raise ValueError('trade_date')

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").reset_index(drop=True)

    required_cols = FEATURE_COLUMNS + [LABEL_5D_COL, LABEL_10D_COL, LABEL_20D_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f": {missing}")

    return df


# =========================================================
# =========================================================
def zscore_per_feature(window_array: np.ndarray) -> np.ndarray:
    'Processing status'
    mean = np.mean(window_array, axis=0, keepdims=True)
    std = np.std(window_array, axis=0, keepdims=True)

    std = np.where(std < 1e-8, 1.0, std)

    normalized = (window_array - mean) / std
    return normalized


# =========================================================
# =========================================================
def build_image_samples(
    df: pd.DataFrame,
    feature_cols: List[str],
    window_size: int = 60,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    '\n    \n    X: shape = (n_samples, n_features, window_size)\n    y_5d: shape = (n_samples,)\n    y_20d: shape = (n_samples,)\n    meta_df: \n'
    X_list = []
    y_5d_list = []
    y_10d_list = []
    y_20d_list = []
    meta_rows = []

    data = df[feature_cols].values

    for end_idx in range(window_size - 1, len(df)):
        start_idx = end_idx - window_size + 1

        label_5d = df.loc[end_idx, LABEL_5D_COL]
        label_10d = df.loc[end_idx, LABEL_10D_COL]
        label_20d = df.loc[end_idx, LABEL_20D_COL]

        if pd.isna(label_5d) or pd.isna(label_10d) or pd.isna(label_20d):
            continue

        window = data[start_idx:end_idx + 1].astype(float)

        if np.isnan(window).any():
            continue

        window_norm = zscore_per_feature(window)

        image_like = window_norm.T

        X_list.append(image_like)
        y_5d_list.append(int(label_5d))
        y_10d_list.append(int(label_10d))
        y_20d_list.append(int(label_20d))

        meta_rows.append({
            "sample_end_idx": end_idx,
            "window_start_date": df.loc[start_idx, "trade_date"].strftime("%Y-%m-%d"),
            "window_end_date": df.loc[end_idx, "trade_date"].strftime("%Y-%m-%d"),
            "label_5d": int(label_5d),
            "label_10d": int(label_10d),
            "label_20d": int(label_20d),
        })

    X = np.array(X_list, dtype=np.float32)
    y_5d = np.array(y_5d_list, dtype=np.int64)
    y_10d = np.array(y_10d_list, dtype=np.int64)
    y_20d = np.array(y_20d_list, dtype=np.int64)
    meta_df = pd.DataFrame(meta_rows)

    return X, y_5d, y_10d, y_20d, meta_df


# =========================================================
# =========================================================
def save_dataset(
    X: np.ndarray,
    y_5d: np.ndarray,
    y_10d: np.ndarray,
    y_20d: np.ndarray,
    meta_df: pd.DataFrame,
    output_npz: Path,
    output_meta: Path,
    feature_cols: List[str],
):
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_npz,
        X=X,
        y_5d=y_5d,
        y_10d=y_10d,
        y_20d=y_20d,
        feature_names=np.array(feature_cols, dtype=object),
    )

    meta_df.to_csv(output_meta, index=False, encoding="utf-8-sig")


# =========================================================
# =========================================================
def main():
    print(f"[INFO] : {INPUT_FILE}")
    df = load_daily_data(INPUT_FILE)

    print(f"[INFO] : {len(df)}")
    print(f"[INFO] : {FEATURE_COLUMNS}")
    print(f"[INFO] : {WINDOW_SIZE}")

    X, y_5d, y_10d, y_20d, meta_df = build_image_samples(
        df=df,
        feature_cols=FEATURE_COLUMNS,
        window_size=WINDOW_SIZE,
    )

    if len(X) == 0:
        raise RuntimeError('Processing status')

    save_dataset(
        X=X,
        y_5d=y_5d,
        y_10d=y_10d,
        y_20d=y_20d,
        meta_df=meta_df,
        output_npz=OUTPUT_NPZ,
        output_meta=OUTPUT_META,
        feature_cols=FEATURE_COLUMNS,
    )

    print('\n===  ===')
    print(f"X shape: {X.shape}")          # (n_samples, n_features, window_size)
    print(f"y_5d shape: {y_5d.shape}")
    print(f"y_10d shape: {y_10d.shape}")
    print(f"y_20d shape: {y_20d.shape}")

    print('\n===  ===')
    unique_5d, counts_5d = np.unique(y_5d, return_counts=True)
    print("label_5d:", dict(zip(unique_5d.tolist(), counts_5d.tolist())))

    unique_10d, counts_10d = np.unique(y_10d, return_counts=True)
    print("label_10d:", dict(zip(unique_10d.tolist(), counts_10d.tolist())))

    unique_20d, counts_20d = np.unique(y_20d, return_counts=True)
    print("label_20d:", dict(zip(unique_20d.tolist(), counts_20d.tolist())))

    print('\n===  ===')
    print(meta_df.head(10))

    print(f"\n[OK] : {OUTPUT_NPZ}")
    print(f"[OK] : {OUTPUT_META}")


if __name__ == "__main__":
    main()
