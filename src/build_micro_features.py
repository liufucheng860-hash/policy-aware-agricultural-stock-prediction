from pathlib import Path
from typing import Dict, List

import pandas as pd


# =========================================================
# =========================================================
EVENT_FILE = Path("data/raw/longping/longping_events.csv")
META_FILE = Path("data/processed/longping/longping_image_metadata.csv")
OUTPUT_FILE = Path("data/processed/longping/longping_micro_features.csv")

ROLLING_WINDOWS = [90, 180]

TRACKED_EVENT_TYPES = [
    "equity_pledge",
    "regulatory_inquiry",
    "shareholder_reduction",
    "private_placement_plan",
    "related_party_transaction",
    "impairment_loss",
    "strategic_cooperation",
    "project_bid_win",
    "contract_signing",
    "product_delivery",
    "supplement_correction",
    "other",
]

VALID_POLARITIES = {"positive", "negative", "neutral"}


# =========================================================
# =========================================================
def load_company_events(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f": {file_path}")

    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"]
    df = None
    last_error = None

    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc, dtype=str, engine="python")
            print(f"[INFO]  company_events.csv={enc}")
            break
        except Exception as e:
            last_error = e

    if df is None:
        raise RuntimeError(f" company_events.csv: {last_error}")

    required_cols = [
        "symbol", "company", "sector",
        "announce_date", "title", "event_type", "event_polarity"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"company_events.csv : {missing}\n: {list(df.columns)}")

    df = df.copy()
    df["announce_date"] = pd.to_datetime(df["announce_date"], errors="coerce")

    if df["announce_date"].isna().any():
        bad = df[df["announce_date"].isna()]
        raise ValueError(f"announce_date \n{bad.head()}")

    df["event_type"] = df["event_type"].astype(str).str.strip()
    df["event_polarity"] = df["event_polarity"].astype(str).str.strip().str.lower()

    valid_polarities = {"positive", "negative", "neutral"}
    bad_pol = df.loc[~df["event_polarity"].isin(valid_polarities), "event_polarity"].unique()
    if len(bad_pol) > 0:
        raise ValueError(f" event_polarity: {bad_pol} {valid_polarities}")

    df = df.sort_values("announce_date").reset_index(drop=True)

    print(f"[INFO] : {len(df)}")
    print('[INFO] event_type')
    print(df["event_type"].value_counts().head(20))
    print('\n[INFO] event_polarity')
    print(df["event_polarity"].value_counts())

    return df


# =========================================================
# =========================================================
def load_metadata(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f": {file_path}")

    meta_df = pd.read_csv(file_path, encoding="utf-8-sig")
    if "window_end_date" not in meta_df.columns:
        raise ValueError('jifeng_image_metadata.csv  window_end_date')

    meta_df = meta_df.copy()
    meta_df["window_end_date"] = pd.to_datetime(meta_df["window_end_date"], errors="coerce")
    if meta_df["window_end_date"].isna().any():
        raise ValueError('window_end_date')

    print(f"[INFO] : {len(meta_df)}")
    return meta_df


# =========================================================
# =========================================================
def days_since_last_event(event_dates: pd.Series, current_date: pd.Timestamp) -> int:
    past_events = event_dates[event_dates <= current_date]
    if len(past_events) == 0:
        return -1
    return int((current_date - past_events.max()).days)


# =========================================================
# =========================================================
def build_features_for_date(events_df: pd.DataFrame, current_date: pd.Timestamp) -> Dict:
    row = {
        "sample_date": current_date.strftime("%Y-%m-%d"),
    }

    # -------------------------
    # -------------------------
    row["days_since_last_any"] = days_since_last_event(events_df["announce_date"], current_date)

    pos_dates = events_df.loc[events_df["event_polarity"] == "positive", "announce_date"]
    neg_dates = events_df.loc[events_df["event_polarity"] == "negative", "announce_date"]
    neu_dates = events_df.loc[events_df["event_polarity"] == "neutral", "announce_date"]

    row["days_since_last_pos"] = days_since_last_event(pos_dates, current_date)
    row["days_since_last_neg"] = days_since_last_event(neg_dates, current_date)
    row["days_since_last_neu"] = days_since_last_event(neu_dates, current_date)

    # -------------------------
    # -------------------------
    same_day_events = events_df[events_df["announce_date"] == current_date]
    row["has_event_today"] = int(len(same_day_events) > 0)
    row["event_count_today"] = int(len(same_day_events))

    # -------------------------
    # -------------------------
    for window in ROLLING_WINDOWS:
        start_date = current_date - pd.Timedelta(days=window)

        window_events = events_df[
            (events_df["announce_date"] <= current_date) &
            (events_df["announce_date"] > start_date)
        ]

        pos_count = int((window_events["event_polarity"] == "positive").sum())
        neg_count = int((window_events["event_polarity"] == "negative").sum())
        neu_count = int((window_events["event_polarity"] == "neutral").sum())

        row[f"all_count_{window}d"] = int(len(window_events))
        row[f"pos_count_{window}d"] = pos_count
        row[f"neg_count_{window}d"] = neg_count
        row[f"neu_count_{window}d"] = neu_count

        row[f"net_sent_{window}d"] = pos_count - neg_count

        row[f"weighted_sent_{window}d"] = pos_count - neg_count

        for etype in TRACKED_EVENT_TYPES:
            row[f"{etype}_{window}d"] = int((window_events["event_type"] == etype).sum())

    return row


# =========================================================
# =========================================================
def build_micro_features(events_df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, meta_row in meta_df.iterrows():
        current_date = meta_row["window_end_date"]

        feature_row = build_features_for_date(events_df, current_date)

        for col in meta_df.columns:
            feature_row[col] = meta_row[col]

        rows.append(feature_row)

    micro_df = pd.DataFrame(rows)

    first_cols = ["sample_date"]
    remain_cols = [c for c in micro_df.columns if c not in first_cols]
    micro_df = micro_df[first_cols + remain_cols]

    return micro_df


# =========================================================
# =========================================================
def save_micro_features(micro_df: pd.DataFrame, output_file: Path):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    micro_df.to_csv(output_file, index=False, encoding="utf-8-sig")


# =========================================================
# =========================================================
def main():
    print(f"[INFO] : {EVENT_FILE}")
    events_df = load_company_events(EVENT_FILE)

    print(f"\n[INFO] : {META_FILE}")
    meta_df = load_metadata(META_FILE)

    micro_df = build_micro_features(events_df, meta_df)

    save_micro_features(micro_df, OUTPUT_FILE)

    print('\n=== 10 ===')
    print(micro_df.head(10).to_string())

    print(f"\n[OK] : {OUTPUT_FILE}")
    print(f"[INFO] : {micro_df.shape}")


if __name__ == "__main__":
    main()
