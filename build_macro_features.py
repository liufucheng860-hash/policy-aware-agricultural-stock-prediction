from pathlib import Path
import pandas as pd


# =========================================================
# =========================================================
OUTPUT_DIR = Path("data/raw")
OUTPUT_FILE = OUTPUT_DIR / "macro_policy_yearly.csv"


# =========================================================
# =========================================================
def build_macro_policy_table() -> pd.DataFrame:
    '\n    \n    \n    - policy_strength: 1~3\n    - machinery_focus: //0~3\n    - smart_agri_focus: //AI+0~3\n    - seed_focus: //0~3\n    - rural_reform_focus: 0~3\n    - subsidy_support: //0~3\n    - technology_focus: /0~3\n'

    rows = [
        # year, policy_strength, machinery_focus, smart_agri_focus, seed_focus,
        # rural_reform_focus, subsidy_support, technology_focus
        [2016, 1, 1, 0, 0, 2, 1, 1],
        [2017, 1, 1, 0, 0, 2, 1, 1],
        [2018, 1, 1, 0, 0, 2, 1, 1],
        [2019, 1, 1, 0, 0, 2, 1, 1],
        [2020, 1, 1, 1, 0, 2, 1, 1],
        [2021, 1, 1, 1, 0, 2, 1, 1],
        [2022, 1, 1, 1, 1, 2, 1, 2],
        [2023, 2, 1, 1, 1, 3, 2, 2],
        [2024, 2, 2, 1, 1, 3, 2, 2],
        [2025, 3, 2, 2, 2, 3, 2, 3],
        [2026, 3, 2, 2, 2, 3, 2, 3],
    ]

    columns = [
        "year",
        "policy_strength",
        "machinery_focus",
        "smart_agri_focus",
        "seed_focus",
        "rural_reform_focus",
        "subsidy_support",
        "technology_focus",
    ]

    df = pd.DataFrame(rows, columns=columns)

    df["macro_total_score"] = (
        df["policy_strength"]
        + df["machinery_focus"]
        + df["smart_agri_focus"]
        + df["seed_focus"]
        + df["rural_reform_focus"]
        + df["subsidy_support"]
        + df["technology_focus"]
    )

    df["jifeng_relevant_score"] = (
        0.25 * df["machinery_focus"]
        + 0.35 * df["smart_agri_focus"]
        + 0.10 * df["seed_focus"]
        + 0.10 * df["subsidy_support"]
        + 0.20 * df["technology_focus"]
    )

    return df


# =========================================================
# =========================================================
def save_macro_policy_table(df: pd.DataFrame, output_file: Path):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")


# =========================================================
# =========================================================
def main():
    df = build_macro_policy_table()
    save_macro_policy_table(df, OUTPUT_FILE)

    print('===  ===')
    print(df)

    print(f"\n[OK] : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
