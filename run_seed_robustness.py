from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs" / "seed_robustness"

SEEDS = [1, 7, 21, 42, 2024]
STOCKS = ["longping", "jifeng", "yituo", "xingguang"]

TASKS = [
    ("image_micro", "Image+Micro", "5d", "train_image_micro_seed.py"),
    ("image_micro", "Image+Micro", "10d", "train_image_micro_seed.py"),
    ("image_macro", "Image+Macro", "20d", "train_image_macro_seed.py"),
    ("full_model", "Full Model", "20d", "train_image_macro_micro_seed.py"),
    ("gated_fusion", "Gated Fusion", "5d", "train_image_macro_micro_gated_seed.py"),
    ("gated_fusion", "Gated Fusion", "10d", "train_image_macro_micro_gated_seed.py"),
    ("gated_fusion", "Gated Fusion", "20d", "train_image_macro_micro_gated_seed.py"),
]

SCRIPT_PATHS = {
    "Image+Micro": ROOT / "train_image_micro.py",
    "Image+Macro": ROOT / "train_image_macro.py",
    "Full Model": ROOT / "train_image_macro_micro.py",
    "Gated Fusion": ROOT / "train_image_macro_micro_gated.py",
}

LATEX_ORDER = [
    ("image_micro", "Image+Micro", "5d"),
    ("image_micro", "Image+Micro", "10d"),
    ("image_macro", "Image+Macro", "20d"),
    ("full_model", "Full Model", "20d"),
    ("gated_fusion", "Gated Fusion", "5d"),
    ("gated_fusion", "Gated Fusion", "10d"),
    ("gated_fusion", "Gated Fusion", "20d"),
]


def run_one(stock: str, model: str, horizon: str, script: str, seed: int) -> bool:
    summary_path = OUTPUT_ROOT / stock / model / horizon / f"metrics_seed{seed}_summary.csv"
    if summary_path.exists():
        print(f"[SKIP] Existing summary: {summary_path}")
        return True

    cmd = [
        sys.executable,
        str(ROOT / script),
        "--stock",
        stock,
        "--horizon",
        horizon,
        "--seed",
        str(seed),
    ]
    print("[RUN]", " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode == 0 and summary_path.exists()


def collect_raw() -> pd.DataFrame:
    rows = []
    for path in sorted(OUTPUT_ROOT.glob("*/*/*/metrics_seed*_summary.csv")):
        rows.append(pd.read_csv(path))
    if rows:
        raw = pd.concat(rows, ignore_index=True)
    else:
        raw = pd.DataFrame(columns=["stock", "model", "horizon", "seed", "accuracy", "f1", "precision", "recall"])

    raw = raw[["stock", "model", "horizon", "seed", "accuracy", "f1", "precision", "recall"]]
    raw = raw.sort_values(["stock", "model", "horizon", "seed"]).reset_index(drop=True)
    raw.to_csv(OUTPUT_ROOT / "seed_robustness_raw.csv", index=False, encoding="utf-8-sig")
    return raw


def build_summary(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        summary = pd.DataFrame(columns=[
            "model", "horizon",
            "mean_accuracy", "std_accuracy",
            "mean_f1", "std_f1",
            "mean_precision", "std_precision",
            "mean_recall", "std_recall",
        ])
    else:
        summary = (
            raw.groupby(["model", "horizon"], as_index=False)
            .agg(
                mean_accuracy=("accuracy", "mean"),
                std_accuracy=("accuracy", "std"),
                mean_f1=("f1", "mean"),
                std_f1=("f1", "std"),
                mean_precision=("precision", "mean"),
                std_precision=("precision", "std"),
                mean_recall=("recall", "mean"),
                std_recall=("recall", "std"),
            )
        )
    summary.to_csv(OUTPUT_ROOT / "seed_robustness_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def write_latex(summary: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Seed robustness results under five random initializations.}",
        r"\label{tab:seed_robustness}",
        r"\begin{tabular}{lcc}",
        r"\hline",
        r"Model & Horizon & F1 Mean $\pm$ Std \\",
        r"\hline",
    ]

    for model, label, horizon in LATEX_ORDER:
        match = summary[(summary["model"] == model) & (summary["horizon"] == horizon)]
        if match.empty:
            value = "N/A"
        else:
            row = match.iloc[0]
            value = f"{row['mean_f1']:.3f} $\\pm$ {row['std_f1']:.3f}"
        lines.append(f"{label} & {horizon} & {value} \\\\")

    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])
    text = "\n".join(lines)
    (OUTPUT_ROOT / "latex_seed_robustness_table.txt").write_text(text, encoding="utf-8")
    return text


def write_conclusion(summary: pd.DataFrame) -> str:
    if summary.empty:
        text = (
            "No completed seed robustness results were available, so no robustness conclusion can be drawn. "
            "The Appendix table should only be included after the missing experiments are completed."
        )
    else:
        max_row = summary.sort_values("std_f1", ascending=False).iloc[0]
        max_std = max_row["std_f1"]
        if max_std <= 0.05:
            stability = "The main conclusions appear stable across the five random initializations."
        elif max_std <= 0.10:
            stability = "The main conclusions are broadly consistent, although some seed-level variation is visible."
        else:
            stability = "The results show noticeable sensitivity to random initialization, so conclusions should be stated cautiously."

        text = (
            f"{stability}\n"
            f"The largest F1 variation is observed for {max_row['model']} at {max_row['horizon']} "
            f"(std = {max_std:.3f}), which should be treated as the least stable setting in this experiment.\n"
            "These results are appropriate for an Appendix seed-robustness table because they summarize repeated runs without overstating statistical certainty.\n"
            "The evidence should be interpreted as a robustness check over the tested stocks and seeds, not as a guarantee of invariance under all random initializations or data splits."
        )

    (OUTPUT_ROOT / "seed_robustness_conclusion.txt").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("Found training scripts:")
    for label, path in SCRIPT_PATHS.items():
        print(f"- {label}: {path}")

    failures = []
    success_count = 0

    print("\nRunning initial test task: longping / Image+Micro / 5d / seed=1")
    if not run_one("longping", "image_micro", "5d", "train_image_micro_seed.py", 1):
        failures.append(("longping", "image_micro", "5d", 1))
        print("[STOP] Initial test task failed. Batch run was not started.")
    else:
        print("[OK] Initial test task succeeded. Starting batch run.")
        for stock in STOCKS:
            for model, _label, horizon, script in TASKS:
                for seed in SEEDS:
                    ok = run_one(stock, model, horizon, script, seed)
                    if ok:
                        success_count += 1
                    else:
                        failures.append((stock, model, horizon, seed))

    raw = collect_raw()
    summary = build_summary(raw)
    latex = write_latex(summary)
    conclusion = write_conclusion(summary)

    print("\nFinal report")
    print(f"Successful experiments: {len(raw)}")
    print(f"Failed tasks: {failures if failures else 'None'}")
    print(f"Raw CSV: {OUTPUT_ROOT / 'seed_robustness_raw.csv'}")
    print(f"Summary CSV: {OUTPUT_ROOT / 'seed_robustness_summary.csv'}")
    print("\nLaTeX table:")
    print(latex)
    print("\nConclusion:")
    print(conclusion)


if __name__ == "__main__":
    main()
