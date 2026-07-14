from pathlib import Path
import pandas as pd


# =========================================================
# =========================================================
INPUT_FILE = Path("data/raw/longping/longping_announcements_raw.xls")
OUTPUT_FILE = Path("data/raw/longping/longping_events.csv")

DEFAULT_SYMBOL = "000998.SZ"
DEFAULT_COMPANY = 'Processing status'
DEFAULT_SECTOR = 'Processing status'

FILTER_NOISY_EVENTS = True


# =========================================================
# =========================================================
EVENT_RULES = [
    ("project_bid_win", ['Processing status', 'Processing status', 'Processing status']),
    ("contract_signing", ['Processing status', 'Processing status', 'Processing status', 'Processing status', 'Processing status']),
    ("strategic_cooperation", ['Processing status', 'Processing status', 'Processing status', 'Processing status', 'Processing status']),
    ("product_delivery", ['Processing status', 'Processing status', 'Processing status', 'Processing status']),
    ("government_subsidy", ['Processing status', 'Processing status', 'Processing status']),
    ("earnings_forecast_positive", ['Processing status', 'Processing status', 'Processing status', 'Processing status']),
    ("shareholder_increase", ['Processing status', 'Processing status']),

    ("earnings_warning_loss", ['Processing status', 'Processing status', 'Processing status', 'Processing status', 'Processing status']),
    ("shareholder_reduction", ['Processing status', 'Processing status', 'Processing status']),
    ("equity_pledge", ['Processing status', 'Processing status', 'Processing status']),
    ("regulatory_inquiry", ['Processing status', 'Processing status', 'Processing status', 'Processing status', 'Processing status', 'Processing status']),
    ("litigation_arbitration", ['Processing status', 'Processing status', 'Processing status', 'Processing status']),
    ("contract_termination_negative", ['Processing status', 'Processing status', 'Processing status', 'Processing status']),
    ("impairment_loss", ['Processing status', 'Processing status', 'Processing status']),
    ("risk_warning", ['Processing status', 'Processing status', 'Processing status']),

    ("private_placement_plan", ['Processing status', 'Processing status', 'Processing status']),
    ("convertible_bond", ['Processing status', 'Processing status']),
    ("board_resolution", ['Processing status', 'Processing status']),
    ("supervisor_resolution", ['Processing status', 'Processing status']),
    ("equity_incentive", ['Processing status', 'Processing status', 'Processing status']),
    ("related_party_transaction", ['Processing status']),
    ("supplement_correction", ['Processing status', 'Processing status', 'Processing status']),
]


POSITIVE_TYPES = {
    "project_bid_win",
    "contract_signing",
    "strategic_cooperation",
    "product_delivery",
    "government_subsidy",
    "earnings_forecast_positive",
    "shareholder_increase",
}

NEGATIVE_TYPES = {
    "earnings_warning_loss",
    "shareholder_reduction",
    "equity_pledge",
    "regulatory_inquiry",
    "litigation_arbitration",
    "contract_termination_negative",
    "impairment_loss",
    "risk_warning",
}

NEUTRAL_TYPES = {
    "private_placement_plan",
    "convertible_bond",
    "board_resolution",
    "supervisor_resolution",
    "equity_incentive",
    "related_party_transaction",
    "supplement_correction",
    "other",
}


# =========================================================
# =========================================================
NOISY_KEYWORDS = [
    'Processing status',
    'Processing status',
    'Processing status',
    'Processing status',
    'Processing status',
    'Processing status',
    'Processing status',
    'Processing status',
    'Processing status',
    'Processing status',
    'Processing status',
    'Processing status',
    'Processing status',
    'Processing status',
    'Processing status',
    'Processing status',
    'Processing status',
    'Processing status',
]


# =========================================================
# =========================================================
def classify_event_type(title: str) -> str:
    if not isinstance(title, str):
        return "other"

    for event_type, keywords in EVENT_RULES:
        for kw in keywords:
            if kw in title:
                return event_type
    return "other"


def infer_polarity(event_type: str, title: str) -> str:
    if event_type in POSITIVE_TYPES:
        return "positive"
    if event_type in NEGATIVE_TYPES:
        return "negative"
    if event_type in NEUTRAL_TYPES:
        return "neutral"

    title = str(title)

    positive_keywords = ['Processing status', 'Processing status', 'Processing status', 'Processing status', 'Processing status', 'Processing status', 'Processing status', 'Processing status', 'Processing status', 'Processing status', 'Processing status']
    negative_keywords = ['Processing status', 'Processing status', 'Processing status', 'Processing status', 'Processing status', 'Processing status', 'Processing status', 'Processing status', 'Processing status', 'Processing status']

    if any(kw in title for kw in positive_keywords):
        return "positive"
    if any(kw in title for kw in negative_keywords):
        return "negative"

    return "neutral"


def is_noisy_title(title: str) -> bool:
    if not isinstance(title, str):
        return True
    return any(kw in title for kw in NOISY_KEYWORDS)


# =========================================================
# =========================================================
def read_input_file(input_file: Path) -> pd.DataFrame:
    if not input_file.exists():
        raise FileNotFoundError(f": {input_file}")

    suffix = input_file.suffix.lower()

    if suffix in [".xls", ".xlsx"]:
        try:
            df = pd.read_excel(input_file, dtype=object)
            print(f"[INFO]  Excel: {input_file.name}")
            return df
        except Exception as e:
            raise RuntimeError(
                f" Excel  .xls xlrdpip install xlrd\n: {e}"
            )

    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"]
    last_error = None

    for enc in encodings:
        try:
            df = pd.read_csv(
                input_file,
                dtype=object,
                encoding=enc,
                engine="python",
                encoding_errors="ignore"
            )
            print(f"[INFO]  CSV={enc}")
            return df
        except Exception as e:
            last_error = e

    raise RuntimeError(f": {last_error}")


# =========================================================
# =========================================================
def build_company_events(input_file: Path, output_file: Path):
    df = read_input_file(input_file)

    print(f"[INFO] : {list(df.columns)}")
    print(f"[INFO] : {len(df)}")

    if len(df.columns) < 2:
        raise ValueError('title  date')

    title_col = df.columns[0]
    date_col = df.columns[1]

    print(f"[INFO] : {title_col}")
    print(f"[INFO] : {date_col}")

    out = pd.DataFrame()
    out["symbol"] = DEFAULT_SYMBOL
    out["company"] = DEFAULT_COMPANY
    out["sector"] = DEFAULT_SECTOR
    out["title"] = df[title_col].astype(str).str.strip()
    out["announce_date"] = pd.to_datetime(df[date_col], errors="coerce")

    out = out.dropna(subset=["announce_date"])
    out = out[out["title"].notna()]
    out = out[out["title"].str.len() > 0].copy()

    if FILTER_NOISY_EVENTS:
        before = len(out)
        out = out[~out["title"].apply(is_noisy_title)].copy()
        after = len(out)
        print(f"[INFO] : {before} -> {after} {before - after} ")

    out["event_type"] = out["title"].apply(classify_event_type)
    out["event_polarity"] = out.apply(
        lambda row: infer_polarity(row["event_type"], row["title"]),
        axis=1
    )

    out = out.sort_values("announce_date").drop_duplicates(
        subset=["announce_date", "title"], keep="first"
    ).reset_index(drop=True)

    out["announce_date"] = out["announce_date"].dt.strftime("%Y-%m-%d")

    out = out[
        ["symbol", "company", "sector", "announce_date", "title", "event_type", "event_polarity"]
    ]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_file, index=False, encoding="utf-8-sig")

    print('\n=== 20 ===')
    print(out.head(20).to_string())

    print('\n=== event_type  ===')
    print(out["event_type"].value_counts(dropna=False))

    print('\n=== event_polarity  ===')
    print(out["event_polarity"].value_counts(dropna=False))

    print(f"\n[OK] : {output_file}")
    print(f"[OK] : {len(out)}")


if __name__ == "__main__":
    build_company_events(INPUT_FILE, OUTPUT_FILE)
