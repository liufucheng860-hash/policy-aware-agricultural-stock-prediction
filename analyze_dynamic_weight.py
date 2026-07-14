from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# =========================================================
# =========================================================
DATA_DIR = Path("outputs/longping")

WEIGHT_5D_FILE = DATA_DIR / "dynamic_weights_5d_sub2021.csv"
WEIGHT_20D_FILE = DATA_DIR / "dynamic_weights_20d_sub2021.csv"

OUTPUT_DIR = DATA_DIR / "dynamic_weight_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

POLICY_DATES = {
    "2021 No.1 Doc": "2021-02-21",
    "2022 No.1 Doc": "2022-02-22",
    "2023 No.1 Doc": "2023-02-13",
    "2024 No.1 Doc": "2024-02-03",
    "2025 No.1 Doc": "2025-02-23",
    "2026 No.1 Doc": "2026-02-03",
}

IMAGE_DOMINANT_TH = 0.60
MACRO_DOMINANT_TH = 0.30
MICRO_DOMINANT_TH = 0.30


# =========================================================
# =========================================================
def calc_metrics(df):
    return {
        "n": len(df),
        "accuracy": accuracy_score(df["y_true"], df["y_pred"]),
        "f1": f1_score(df["y_true"], df["y_pred"], zero_division=0),
        "precision": precision_score(df["y_true"], df["y_pred"], zero_division=0),
        "recall": recall_score(df["y_true"], df["y_pred"], zero_division=0),
    }


def load_weight_file(file_path):
    df = pd.read_csv(file_path, encoding="utf-8-sig")

    date_col_candidates = ["window_end_date", "date", "trade_date"]
    date_col = None
    for c in date_col_candidates:
        if c in df.columns:
            date_col = c
            break

    if date_col is None:
        raise ValueError(f": {df.columns.tolist()}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "date"})
    df = df.sort_values("date").reset_index(drop=True)

    required_cols = ["w_image", "w_macro", "w_micro", "y_true", "y_pred", "y_prob"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f": {missing}")

    return df


# =========================================================
# =========================================================
def plot_weight_time_series(df, horizon):
    plt.figure(figsize=(14, 6))

    plt.plot(df["date"], df["w_image"], label="image weight", linewidth=1.5)
    plt.plot(df["date"], df["w_macro"], label="macro weight", linewidth=1.5)
    plt.plot(df["date"], df["w_micro"], label="micro weight", linewidth=1.5)

    for name, d in POLICY_DATES.items():
        d = pd.to_datetime(d)
        if df["date"].min() <= d <= df["date"].max():
            plt.axvline(d, linestyle="--", linewidth=1)
            plt.text(d, 1.02, name, rotation=90, fontsize=8, va="bottom")

    plt.title(f"Dynamic Modality Weights Over Time - {horizon}")
    plt.xlabel("Date")
    plt.ylabel("Weight")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.tight_layout()

    out_file = OUTPUT_DIR / f"dynamic_weights_time_series_{horizon}.png"
    plt.savefig(out_file, dpi=300)
    plt.close()

    print(f"[OK] saved: {out_file}")


# =========================================================
# =========================================================
def plot_weight_histogram(df, horizon):
    for col in ["w_image", "w_macro", "w_micro"]:
        plt.figure(figsize=(8, 5))
        plt.hist(df[col], bins=30)
        plt.title(f"{col} Distribution - {horizon}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()

        out_file = OUTPUT_DIR / f"{col}_hist_{horizon}.png"
        plt.savefig(out_file, dpi=300)
        plt.close()

        print(f"[OK] saved: {out_file}")


# =========================================================
# =========================================================
def assign_dominant_modality(row):
    weights = {
        "image": row["w_image"],
        "macro": row["w_macro"],
        "micro": row["w_micro"],
    }
    return max(weights, key=weights.get)


def analyze_by_dominant_modality(df, horizon):
    df = df.copy()
    df["dominant_modality"] = df.apply(assign_dominant_modality, axis=1)

    rows = []
    for g, sub in df.groupby("dominant_modality"):
        m = calc_metrics(sub)
        rows.append({
            "horizon": horizon,
            "group_type": "dominant_modality",
            "group": g,
            "n": m["n"],
            "accuracy": m["accuracy"],
            "f1": m["f1"],
            "precision": m["precision"],
            "recall": m["recall"],
            "w_image_mean": sub["w_image"].mean(),
            "w_macro_mean": sub["w_macro"].mean(),
            "w_micro_mean": sub["w_micro"].mean(),
        })

    result = pd.DataFrame(rows)
    out_file = OUTPUT_DIR / f"dominant_modality_metrics_{horizon}.csv"
    result.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"[OK] saved: {out_file}")
    return result


# =========================================================
# =========================================================
def analyze_by_weight_uncertainty(df, horizon):
    df = df.copy()

    eps = 1e-12
    weights = df[["w_image", "w_macro", "w_micro"]].values
    entropy = -np.sum(weights * np.log(weights + eps), axis=1)
    df["weight_entropy"] = entropy

    median_entropy = df["weight_entropy"].median()
    df["weight_regime"] = np.where(
        df["weight_entropy"] >= median_entropy,
        "high_entropy_balanced",
        "low_entropy_dominant"
    )

    rows = []
    for g, sub in df.groupby("weight_regime"):
        m = calc_metrics(sub)
        rows.append({
            "horizon": horizon,
            "group_type": "weight_entropy_regime",
            "group": g,
            "n": m["n"],
            "accuracy": m["accuracy"],
            "f1": m["f1"],
            "precision": m["precision"],
            "recall": m["recall"],
            "entropy_mean": sub["weight_entropy"].mean(),
            "w_image_mean": sub["w_image"].mean(),
            "w_macro_mean": sub["w_macro"].mean(),
            "w_micro_mean": sub["w_micro"].mean(),
        })

    result = pd.DataFrame(rows)
    out_file = OUTPUT_DIR / f"weight_entropy_regime_metrics_{horizon}.csv"
    result.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"[OK] saved: {out_file}")
    return result


# =========================================================
# =========================================================
def analyze_policy_windows(df, horizon, window_days=20):
    df = df.copy()
    df["near_policy"] = False

    for _, d in POLICY_DATES.items():
        d = pd.to_datetime(d)
        start = d - pd.Timedelta(days=window_days)
        end = d + pd.Timedelta(days=window_days)
        df.loc[(df["date"] >= start) & (df["date"] <= end), "near_policy"] = True

    rows = []
    for g, sub in df.groupby("near_policy"):
        label = f"policy_window_{window_days}d" if g else "non_policy_window"
        m = calc_metrics(sub)
        rows.append({
            "horizon": horizon,
            "group_type": "policy_window",
            "group": label,
            "n": m["n"],
            "accuracy": m["accuracy"],
            "f1": m["f1"],
            "precision": m["precision"],
            "recall": m["recall"],
            "w_image_mean": sub["w_image"].mean(),
            "w_macro_mean": sub["w_macro"].mean(),
            "w_micro_mean": sub["w_micro"].mean(),
        })

    result = pd.DataFrame(rows)
    out_file = OUTPUT_DIR / f"policy_window_metrics_{horizon}.csv"
    result.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"[OK] saved: {out_file}")
    return result


# =========================================================
# =========================================================
def analyze_one(file_path, horizon):
    print(f"\n========== Analyze {horizon} ==========")
    df = load_weight_file(file_path)

    print("[INFO] basic weight means:")
    print(df[["w_image", "w_macro", "w_micro"]].mean())

    plot_weight_time_series(df, horizon)
    plot_weight_histogram(df, horizon)

    res1 = analyze_by_dominant_modality(df, horizon)
    res2 = analyze_by_weight_uncertainty(df, horizon)
    res3 = analyze_policy_windows(df, horizon, window_days=20)

    all_res = pd.concat([res1, res2, res3], axis=0, ignore_index=True)

    out_file = OUTPUT_DIR / f"all_group_analysis_{horizon}.csv"
    all_res.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"[OK] saved: {out_file}")

    return all_res


def main():
    all_results = []

    if WEIGHT_5D_FILE.exists():
        all_results.append(analyze_one(WEIGHT_5D_FILE, "5d"))
    else:
        print(f"[WARN] missing: {WEIGHT_5D_FILE}")

    if WEIGHT_20D_FILE.exists():
        all_results.append(analyze_one(WEIGHT_20D_FILE, "20d"))
    else:
        print(f"[WARN] missing: {WEIGHT_20D_FILE}")

    if all_results:
        final = pd.concat(all_results, axis=0, ignore_index=True)
        final_file = OUTPUT_DIR / "dynamic_weight_final_summary.csv"
        final.to_csv(final_file, index=False, encoding="utf-8-sig")
        print(f"\n[OK] final summary saved: {final_file}")


if __name__ == "__main__":
    main()
