import time
import random
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tushare as ts


# =========================================================
# =========================================================
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN")

TS_CODE = "603789.SH"
START_DATE = "20160101"
END_DATE = "20260413"

OUTPUT_DIR = Path("data/raw/xingguang")
OUTPUT_FILE = OUTPUT_DIR / "xingguang_daily.csv"


# =========================================================
# =========================================================
if not TUSHARE_TOKEN:
    raise ValueError("Please set the TUSHARE_TOKEN environment variable before running this script.")

ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()


# =========================================================
# =========================================================
def fetch_daily_data(
    ts_code: str,
    start_date: str,
    end_date: str,
    retry: int = 5,
    sleep_sec: float = 1.0,
) -> Optional[pd.DataFrame]:
    '\n     Tushare daily \n'
    for i in range(retry):
        try:
            df = pro.daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
            )
            if df is None or df.empty:
                print(f"[WARN] {ts_code} ")
                return None

            df = df.copy()
            df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
            df = df.sort_values("trade_date").reset_index(drop=True)

            print(f"[INFO]  {ts_code}  {len(df)} ")
            return df

        except Exception as e:
            print(f"[WARN]  {ts_code}  {i+1} : {e}")
            time.sleep(sleep_sec * (i + 1) + random.uniform(0.2, 0.8))

    print(f"[ERROR]  {ts_code} ")
    return None


# =========================================================
# =========================================================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    '\n    \n'
    out = df.copy()

    # ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount
    out = out.rename(columns={"vol": "volume"})

    # ======================
    # ======================
    out["ma5"] = out["close"].rolling(5).mean()
    out["ma20"] = out["close"].rolling(20).mean()
    out["ma60"] = out["close"].rolling(60).mean()

    out["ret_1d"] = out["close"].pct_change()

    out["log_ret_1d"] = np.log(out["close"] / out["close"].shift(1))

    out["volatility_5d"] = out["ret_1d"].rolling(5).std()
    out["volatility_20d"] = out["ret_1d"].rolling(20).std()

    out["volume_ma5"] = out["volume"].rolling(5).mean()
    out["volume_ma20"] = out["volume"].rolling(20).mean()

    # ======================
    # ======================
    out["future_5d_ret"] = out["close"].shift(-5) / out["close"] - 1.0
    out["future_10d_ret"] = out["close"].shift(-10) / out["close"] - 1.0
    out["future_20d_ret"] = out["close"].shift(-20) / out["close"] - 1.0

    out["label_5d"] = (out["future_5d_ret"] > 0).astype("Int64")
    out["label_10d"] = (out["future_10d_ret"] > 0).astype("Int64")
    out["label_20d"] = (out["future_20d_ret"] > 0).astype("Int64")

    # ======================
    # ======================
    out["close_ma5_ratio"] = out["close"] / out["ma5"] - 1.0
    out["close_ma20_ratio"] = out["close"] / out["ma20"] - 1.0
    out["close_ma60_ratio"] = out["close"] / out["ma60"] - 1.0

    rolling_min_20 = out["low"].rolling(20).min()
    rolling_max_20 = out["high"].rolling(20).max()
    out["price_pos_20d"] = (out["close"] - rolling_min_20) / (rolling_max_20 - rolling_min_20)

    return out


# =========================================================
# =========================================================
def finalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    '\n    \n'
    cols = [
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "change",
        "pct_chg",
        "volume",
        "amount",

        "ma5",
        "ma20",
        "ma60",

        "ret_1d",
        "log_ret_1d",
        "volatility_5d",
        "volatility_20d",
        "volume_ma5",
        "volume_ma20",

        "close_ma5_ratio",
        "close_ma20_ratio",
        "close_ma60_ratio",
        "price_pos_20d",

        "future_5d_ret",
        "future_10d_ret",
        "future_20d_ret",
        "label_5d",
        "label_10d",
        "label_20d",
    ]

    existing_cols = [c for c in cols if c in df.columns]
    out = df[existing_cols].copy()

    out = out.dropna().reset_index(drop=True)

    return out


# =========================================================
# =========================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = fetch_daily_data(
        ts_code=TS_CODE,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    if df_raw is None or df_raw.empty:
        raise RuntimeError('Processing status')

    df_feat = build_features(df_raw)
    df_final = finalize_dataset(df_feat)

    df_final.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print('\n=== 10 ===')
    print(df_final.head(10))

    print('\n===  ===')
    print("label_5d:")
    print(df_final["label_5d"].value_counts(dropna=False))
    print("\nlabel_10d:")
    print(df_final["label_10d"].value_counts(dropna=False))
    print("\nlabel_20d:")
    print(df_final["label_20d"].value_counts(dropna=False))

    print(f"\n[OK] : {OUTPUT_FILE}")
    print(f"[INFO] : {len(df_final)}")
    print(f"[INFO] : {df_final['trade_date'].min()} ~ {df_final['trade_date'].max()}")


if __name__ == "__main__":
    main()
