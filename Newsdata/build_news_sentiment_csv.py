from __future__ import annotations

from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


BASE_DIR = Path(__file__).resolve().parent
COMBINED_NEWS_PATH = BASE_DIR / "news_trump_combined.csv"
OUT_PATH = BASE_DIR / "sentiment_2025-11-14_to_2026-04-10.csv"
START_DATE = pd.Timestamp("2025-11-14")
END_DATE = pd.Timestamp("2026-04-10")


def load_combined_news(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["date", "title", "content"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path.name}: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    df = df.dropna(subset=["date"])

    return df


def compute_daily_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()

    text = (
        df["title"].fillna("").astype(str).str.strip()
        + ". "
        + df["content"].fillna("").astype(str).str.strip()
    )

    df = df.copy()
    df["sentiment_article"] = text.apply(lambda x: analyzer.polarity_scores(x)["compound"])

    daily = (
        df.groupby("date", as_index=False)
        .agg(sentiment_mean=("sentiment_article", "mean"))
        .rename(columns={"date": "post_date"})
        .sort_values("post_date")
        .reset_index(drop=True)
    )

    daily["sentiment_roll7"] = daily["sentiment_mean"].rolling(7, min_periods=1).mean()
    return daily


def main() -> None:
    news = load_combined_news(COMBINED_NEWS_PATH)
    news = news[(news["date"] >= START_DATE) & (news["date"] <= END_DATE)].copy()

    daily = compute_daily_sentiment(news)
    export = daily[["post_date", "sentiment_mean", "sentiment_roll7"]]

    export.to_csv(OUT_PATH, index=False)

    print(f"Saved: {OUT_PATH}")
    print(f"Rows : {len(export):,}")
    print(f"Date range: {export['post_date'].min().date()} -> {export['post_date'].max().date()}")


if __name__ == "__main__":
    main()
