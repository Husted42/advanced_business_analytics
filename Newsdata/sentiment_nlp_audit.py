from __future__ import annotations

from pathlib import Path
import argparse
import re
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception as exc:
    raise ImportError(
        "vaderSentiment is required for word-level sentiment contribution analysis. "
        "Install with: python -m pip install vaderSentiment"
    ) from exc


DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "data" / "trump_daily_features.csv"
DEFAULT_GUARDIAN = Path(__file__).resolve().parent / "guardian_trump_articles.csv"
DEFAULT_NYT = Path(__file__).resolve().parent / "nyt_trump_last_6_months.csv"
DEFAULT_START_DATE = "2025-11-14"
DEFAULT_END_DATE = "2026-04-10"
SENTIMENT_COLS = ["sentiment_mean", "sentiment_std", "sentiment_pct_negative", "n_articles"]
TOKEN_PATTERN = re.compile(r"[a-z']+")


def load_news_source(csv_path: Path, source_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = ["date", "title", "content"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns in {csv_path.name}: {missing}"
        )

    df = df[required].copy()
    df["source"] = source_name
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    df = df.dropna(subset=["date"])

    return df


def load_news_data(guardian_path: Path, nyt_path: Path) -> pd.DataFrame:
    guardian = load_news_source(guardian_path, "guardian")
    nyt = load_news_source(nyt_path, "nyt")
    return pd.concat([guardian, nyt], ignore_index=True)


def compute_daily_sentiment(news_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    analyzer = SentimentIntensityAnalyzer()

    text = (
        news_df["title"].fillna("").astype(str).str.strip()
        + ". "
        + news_df["content"].fillna("").astype(str).str.strip()
    )

    article_df = news_df.copy()
    article_df["text"] = text
    article_df["sentiment_article"] = article_df["text"].apply(
        lambda x: analyzer.polarity_scores(x)["compound"] if x else 0.0
    )

    daily_df = (
        article_df.groupby("date", as_index=False)
        .agg(
            sentiment_mean=("sentiment_article", "mean"),
            sentiment_std=("sentiment_article", "std"),
            sentiment_pct_negative=("sentiment_article", lambda x: (x < -0.5).mean()),
            n_articles=("sentiment_article", "size"),
        )
        .rename(columns={"date": "post_date"})
        .sort_values("post_date")
        .reset_index(drop=True)
    )
    daily_df["sentiment_std"] = daily_df["sentiment_std"].fillna(0.0)

    return daily_df, article_df


def filter_date_range(
    daily_df: pd.DataFrame,
    article_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")

    daily_f = daily_df[
        (daily_df["post_date"] >= start_date) & (daily_df["post_date"] <= end_date)
    ].copy()
    posts_f = article_df[
        (article_df["date"] >= start_date) & (article_df["date"] <= end_date)
    ].copy()

    if daily_f.empty:
        raise ValueError(
            "No daily rows found inside selected date range "
            f"({start_date.date()} -> {end_date.date()})."
        )

    return daily_f, posts_f


def aggregate_sentiment_words(texts: pd.Series, positive: bool) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()
    counts: dict[str, int] = defaultdict(int)
    contribution: dict[str, float] = defaultdict(float)

    for text in texts.dropna().astype(str):
        for token in TOKEN_PATTERN.findall(text.lower()):
            score = analyzer.lexicon.get(token)
            if score is None or score == 0:
                continue
            if positive and score > 0:
                counts[token] += 1
                contribution[token] += score
            elif not positive and score < 0:
                counts[token] += 1
                contribution[token] += score

    word_df = pd.DataFrame(
        {
            "word": list(counts.keys()),
            "count": [counts[w] for w in counts],
            "total_lexicon_contribution": [contribution[w] for w in counts],
        }
    )

    if word_df.empty:
        return word_df

    if positive:
        word_df = word_df.sort_values(
            ["total_lexicon_contribution", "count"], ascending=[False, False]
        )
    else:
        word_df = word_df.sort_values(
            ["total_lexicon_contribution", "count"], ascending=[True, False]
        )

    return word_df.reset_index(drop=True)


def print_top_sentiment_words(
    daily_df: pd.DataFrame,
    article_df: pd.DataFrame,
    top_days: int,
    top_words: int,
) -> None:
    pos_days = daily_df.nlargest(top_days, "sentiment_mean")["post_date"]
    neg_days = daily_df.nsmallest(top_days, "sentiment_mean")["post_date"]

    pos_texts = article_df[article_df["date"].isin(pos_days)]["text"]
    neg_texts = article_df[article_df["date"].isin(neg_days)]["text"]

    pos_words = aggregate_sentiment_words(pos_texts, positive=True).head(top_words)
    neg_words = aggregate_sentiment_words(neg_texts, positive=False).head(top_words)

    print("\nTop positive days included:")
    print(
        daily_df[daily_df["post_date"].isin(pos_days)][["post_date", "sentiment_mean", "n_articles"]]
        .sort_values("sentiment_mean", ascending=False)
        .to_string(index=False)
    )

    print("\nTop negative days included:")
    print(
        daily_df[daily_df["post_date"].isin(neg_days)][["post_date", "sentiment_mean", "n_articles"]]
        .sort_values("sentiment_mean", ascending=True)
        .to_string(index=False)
    )

    print(f"\nTop {top_words} positive words contributing to top {top_days} positive days:")
    if pos_words.empty:
        print("No positive lexicon matches found.")
    else:
        print(pos_words.to_string(index=False))

    print(f"\nTop {top_words} negative words contributing to top {top_days} negative days:")
    if neg_words.empty:
        print("No negative lexicon matches found.")
    else:
        print(neg_words.to_string(index=False))


def save_sentiment_plot(df: pd.DataFrame, output_path: Path) -> None:
    window = df.sort_values("post_date").copy()
    window["sentiment_roll7"] = window["sentiment_mean"].rolling(7, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        window["post_date"],
        window["sentiment_mean"],
        label="Daily sentiment_mean",
        linewidth=1.6,
        alpha=0.65,
    )
    ax.plot(
        window["post_date"],
        window["sentiment_roll7"],
        label="7-day rolling mean",
        linewidth=2.4,
    )

    ax.axhline(0, color="grey", linestyle="--", linewidth=1)
    ax.set_title(
        "News daily sentiment — "
        f"{window['post_date'].min().date()} to {window['post_date'].max().date()}"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Sentiment (compound)")
    ax.legend()
    ax.grid(alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"\nSaved sentiment plot to: {output_path}")


def save_sentiment_csv(df: pd.DataFrame, output_path: Path) -> None:
    export_df = df.sort_values("post_date")[["post_date", "sentiment_mean"]].copy()
    export_df["sentiment_roll7"] = export_df["sentiment_mean"].rolling(7, min_periods=1).mean()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(output_path, index=False)

    print(f"Saved sentiment CSV to : {output_path}")


def print_report(df: pd.DataFrame, article_df: pd.DataFrame, top_days: int, top_words: int) -> None:
    print("=" * 70)
    print("NLP Sentiment Audit")
    print("=" * 70)
    print(f"Rows (days): {len(df):,}")
    print(f"Date range : {df['post_date'].min().date()} -> {df['post_date'].max().date()}")

    print("\nSentiment columns:")
    for col in SENTIMENT_COLS:
        non_null_pct = df[col].notna().mean() * 100
        print(f"- {col:<24} non-null: {non_null_pct:6.2f}%")

    summary = df[SENTIMENT_COLS].describe().T[["mean", "std", "min", "max"]]
    print("\nSummary statistics:")
    print(summary.round(4))

    top_pos = df.nlargest(5, "sentiment_mean")[["post_date", "n_articles", "sentiment_mean"]]
    top_neg = df.nsmallest(5, "sentiment_mean")[["post_date", "n_articles", "sentiment_mean"]]

    print("\nTop 5 most positive days (sentiment_mean):")
    print(top_pos.to_string(index=False))

    print("\nTop 5 most negative days (sentiment_mean):")
    print(top_neg.to_string(index=False))

    corr_with_volume = (
        df[["n_articles", "sentiment_mean", "sentiment_std", "sentiment_pct_negative"]]
        .corr(numeric_only=True)
        .loc["n_articles", ["sentiment_mean", "sentiment_std", "sentiment_pct_negative"]]
        .round(4)
    )
    print("\nCorrelation with n_articles:")
    print(corr_with_volume)

    rolling = df.sort_values("post_date").set_index("post_date")["sentiment_mean"].rolling(7).mean()
    if not rolling.dropna().empty:
        latest = rolling.dropna().tail(5)
        print("\nLatest 5 values of 7-day rolling sentiment_mean:")
        print(latest.round(4).to_string())

    print("\n" + "=" * 70)
    print("Word-level sentiment drivers")
    print("=" * 70)
    print_top_sentiment_words(df, article_df, top_days=top_days, top_words=top_words)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect sentiment features from Guardian + NYT news datasets"
    )
    parser.add_argument(
        "--guardian-csv",
        type=Path,
        default=DEFAULT_GUARDIAN,
        help="Path to Guardian dataset (default: Newsdata/guardian_trump_articles.csv)",
    )
    parser.add_argument(
        "--nyt-csv",
        type=Path,
        default=DEFAULT_NYT,
        help="Path to NYT dataset (default: Newsdata/nyt_trump_last_6_months.csv)",
    )
    parser.add_argument(
        "--top-days",
        type=int,
        default=5,
        help="How many most positive and most negative days to include.",
    )
    parser.add_argument(
        "--top-words",
        type=int,
        default=15,
        help="How many words to print for each side.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=DEFAULT_START_DATE,
        help="Inclusive start date for analysis window (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=DEFAULT_END_DATE,
        help="Inclusive end date for analysis window (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path(__file__).resolve().parent / "news_sentiment_2025-11-14_to_2026-04-10.png",
        help="Path for saved sentiment plot PNG.",
    )
    parser.add_argument(
        "--sentiment-csv-output",
        type=Path,
        default=Path(__file__).resolve().parent / "sentiment_2025-11-14_to_2026-04-10.csv",
        help="Path for saved sentiment results CSV.",
    )
    args = parser.parse_args()

    news_df = load_news_data(args.guardian_csv, args.nyt_csv)
    df, article_df = compute_daily_sentiment(news_df)
    start_date = pd.Timestamp(args.start_date)
    end_date = pd.Timestamp(args.end_date)

    df, article_df = filter_date_range(df, article_df, start_date=start_date, end_date=end_date)
    print_report(df, article_df, top_days=args.top_days, top_words=args.top_words)
    save_sentiment_csv(df, output_path=args.sentiment_csv_output)
    save_sentiment_plot(df, output_path=args.plot_output)


if __name__ == "__main__":
    main()
