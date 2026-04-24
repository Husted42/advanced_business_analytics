from __future__ import annotations

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
NYT_PATH = BASE_DIR / "nyt_trump_last_6_months.csv"
GUARDIAN_PATH = BASE_DIR / "guardian_trump_articles.csv"
OUT_PATH = BASE_DIR / "news_trump_combined.csv"


REQUIRED_COLS = ["date", "title", "content"]


def load_source(path: Path, source_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    df = df[REQUIRED_COLS].copy()
    df["source"] = source_name

    # Normalize dates to YYYY-MM-DD where possible
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.date.astype("string")

    return df


def main() -> None:
    nyt = load_source(NYT_PATH, "nyt")
    guardian = load_source(GUARDIAN_PATH, "guardian")

    combined = pd.concat([nyt, guardian], ignore_index=True)
    combined = (
        combined.dropna(subset=["date", "title", "content"])
        .drop_duplicates(subset=["date", "title", "content", "source"])
        .sort_values(["date", "source", "title"])
        .reset_index(drop=True)
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_PATH, index=False)

    print(f"Saved: {OUT_PATH}")
    print(f"Rows : {len(combined):,}")
    print("By source:")
    print(combined["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
