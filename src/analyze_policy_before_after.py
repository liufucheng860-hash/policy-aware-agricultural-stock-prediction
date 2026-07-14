from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# =========================================================
# =========================================================
FILE_PATH = Path("outputs/longping/dynamic_weights_20d_sub2021_ALL.csv")
POLICY_DATE = "2025-02-23"

WINDOW_LIST = [30, 60, 90]

WEIGHT_COLS = ["w_image", "w_macro", "w_micro"]

OUTPUT_FILE = Path("outputs/longping/policy_window_analysis_30_60_90.csv")


# =========================================================
# =========================================================
df = pd.read_csv(FILE_PATH, encoding="utf-8-sig")
df["window_end_date"] = pd.to_datetime(df["window_end_date"])

policy_date = pd.to_datetime(POLICY_DATE)

summary = []

# =========================================================
# =========================================================
for window_days in WINDOW_LIST:
    before_start = policy_date - pd.Timedelta(days=window_days)
    before_end = policy_date

    after_start = policy_date
    after_end = policy_date + pd.Timedelta(days=window_days)

    df_before = df[
        (df["window_end_date"] >= before_start)
        & (df["window_end_date"] < before_end)
    ].copy()

    df_after = df[
        (df["window_end_date"] >= after_start)
        & (df["window_end_date"] <= after_end)
    ].copy()

    print(f"\n================ WINDOW: {window_days} DAYS ================")
    print(f"Before samples: {len(df_before)}")
    print(f"After samples : {len(df_after)}")

    for col in WEIGHT_COLS:
        before_mean = df_before[col].mean()
        after_mean = df_after[col].mean()
        diff = after_mean - before_mean

        if len(df_before) > 1 and len(df_after) > 1:
            stat, p = ttest_ind(
                df_before[col],
                df_after[col],
                equal_var=False,
                nan_policy="omit"
            )
        else:
            stat, p = np.nan, np.nan

        print(f"\n[{col}]")
        print(f"before mean = {before_mean:.4f}")
        print(f"after  mean = {after_mean:.4f}")
        print(f"diff        = {diff:.4f}")
        print(f"t-stat      = {stat:.4f}")
        print(f"p-value     = {p:.6f}")

        summary.append({
            "window_days": window_days,
            "modality": col,
            "before_mean": before_mean,
            "after_mean": after_mean,
            "difference": diff,
            "t_stat": stat,
            "p_value": p,
            "before_n": len(df_before),
            "after_n": len(df_after),
        })

summary_df = pd.DataFrame(summary)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
summary_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print(f"\n[OK] saved: {OUTPUT_FILE}")
