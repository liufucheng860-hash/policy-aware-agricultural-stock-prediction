from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# =========================================================
FILE_PATH = Path("outputs/jifeng/dynamic_weights_20d_sub2021_ALL.csv")
OUTPUT_DIR = Path("outputs/jifeng/policy_aligned_analysis")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for WINDOW in [30, 60, 90]:

    POLICY_DATES = {
    "2021": "2021-02-21",
    "2022": "2022-02-22",
    "2023": "2023-02-13",
    "2024": "2024-02-03",
    "2025": "2025-02-23",
    "2026": "2026-02-23",
    }

# =========================================================
# =========================================================
    df = pd.read_csv(FILE_PATH, encoding="utf-8-sig")

    df["window_end_date"] = pd.to_datetime(df["window_end_date"])

    print("[INFO] date range:")
    print(df["window_end_date"].min())
    print(df["window_end_date"].max())

# =========================================================
# =========================================================
    aligned_rows = []

    for year, date_str in POLICY_DATES.items():

        policy_date = pd.to_datetime(date_str)

        temp = df.copy()

        temp["event_day"] = (
            temp["window_end_date"] - policy_date
        ).dt.days

        temp = temp[
            (temp["event_day"] >= -WINDOW)
            & (temp["event_day"] <= WINDOW)
        ].copy()

        if len(temp) == 0:
            print(f"[SKIP] {year} no samples")
            continue

        temp["policy_year"] = year

        aligned_rows.append(temp)

# =========================================================
# =========================================================
    aligned_df = pd.concat(aligned_rows, axis=0, ignore_index=True)

    print("\n[INFO] aligned sample size:")
    print(len(aligned_df))

# =========================================================
# =========================================================
    agg = aligned_df.groupby("event_day").agg({
    "w_image": ["mean", "std"],
    "w_macro": ["mean", "std"],
    "w_micro": ["mean", "std"],
    }).reset_index()

    agg.columns = [
    "event_day",
    "w_image_mean",
    "w_image_std",
    "w_macro_mean",
    "w_macro_std",
    "w_micro_mean",
    "w_micro_std",
]

# =========================================================
# =========================================================
    csv_file = OUTPUT_DIR / f"policy_aligned_weight_summary_{WINDOW}d.csv"

    agg.to_csv(csv_file, index=False, encoding="utf-8-sig")

    print(f"[OK] saved: {csv_file}")

# =========================================================
# =========================================================
    plt.figure(figsize=(14, 7))

# image
    plt.plot(
        agg["event_day"],
        agg["w_image_mean"],
        label="Image",
        linewidth=2,
    )

# macro
    plt.plot(
        agg["event_day"],
        agg["w_macro_mean"],
        label="Macro",
        linewidth=2,
    )

# micro
    plt.plot(
        agg["event_day"],
        agg["w_micro_mean"],
        label="Micro",
        linewidth=2,
    )

    plt.axvline(
        0,
        linestyle="--",
        linewidth=2,
    )

    plt.text(
        0,
        1.02,
        '',
        fontsize=10,
        ha="center",
    )

    plt.ylim(0, 1.05)

    plt.legend()

    plt.tight_layout()

    png_file = OUTPUT_DIR / f"policy_aligned_weights_{WINDOW}d.png"

    plt.savefig(png_file, dpi=300)

    plt.close()

    print(f"[OK] saved: {png_file}")

# =========================================================
# =========================================================
    before = agg[
    (agg["event_day"] >= -20)
    & (agg["event_day"] < 0)
]

    after = agg[
    (agg["event_day"] > 0)
    & (agg["event_day"] <= 20)
]

    print("\n========== BEFORE POLICY ==========")

    print(before[
    [
        "w_image_mean",
        "w_macro_mean",
        "w_micro_mean",
    ]
].mean())

    print("\n========== AFTER POLICY ==========")

    print(after[
    [
        "w_image_mean",
        "w_macro_mean",
        "w_micro_mean",
    ]
].mean())
