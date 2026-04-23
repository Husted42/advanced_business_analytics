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
DEFAULT_POSTS = Path(__file__).resolve().parents[1] / "data" / "trump_clean_posts.csv"
SENTIMENT_COLS = ["sentiment_mean", "sentiment_std", "sentiment_pct_negative"]
TOKEN_PATTERN = re.compile(r"[a-z']+")


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["post_date"])
    missing = [c for c in SENTIMENT_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected sentiment columns in {csv_path.name}: {missing}"
        )
    return df


def load_posts(csv_path: Path) -> pd.DataFrame:
    posts = pd.read_csv(csv_path, parse_dates=["post_date"])
    required = ["post_date", "text"]
    missing = [c for c in required if c not in posts.columns]
    if missing:
        raise ValueError(f"Missing expected post columns in {csv_path.name}: {missing}")
    return posts


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
    posts_df: pd.DataFrame,
    top_days: int,
    top_words: int,
) -> None:
    pos_days = daily_df.nlargest(top_days, "sentiment_mean")["post_date"]
    neg_days = daily_df.nsmallest(top_days, "sentiment_mean")["post_date"]

    pos_texts = posts_df[posts_df["post_date"].isin(pos_days)]["text"]
    neg_texts = posts_df[posts_df["post_date"].isin(neg_days)]["text"]

    pos_words = aggregate_sentiment_words(pos_texts, positive=True).head(top_words)
    neg_words = aggregate_sentiment_words(neg_texts, positive=False).head(top_words)

    print("\nTop positive days included:")
    print(
        daily_df[daily_df["post_date"].isin(pos_days)][["post_date", "sentiment_mean", "post_count"]]
        .sort_values("sentiment_mean", ascending=False)
        .to_string(index=False)
    )

    print("\nTop negative days included:")
    print(
        daily_df[daily_df["post_date"].isin(neg_days)][["post_date", "sentiment_mean", "post_count"]]
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


def save_sentiment_plot(df: pd.DataFrame, months: int, output_path: Path) -> None:
    if months <= 0:
        raise ValueError("months must be >= 1")

    daily = df.sort_values("post_date").copy()
    max_date = daily["post_date"].max()
    cutoff = max_date - pd.DateOffset(months=months)
    window = daily[daily["post_date"] >= cutoff].copy()

    if window.empty:
        print("\nNo rows found in requested plot window; skipping plot generation.")
        return

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
    ax.set_title(f"Trump daily sentiment — last {months} months")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sentiment (compound)")
    ax.legend()
    ax.grid(alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"\nSaved sentiment plot to: {output_path}")


def print_report(df: pd.DataFrame, posts_df: pd.DataFrame, top_days: int, top_words: int) -> None:
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

    top_pos = df.nlargest(5, "sentiment_mean")[["post_date", "post_count", "sentiment_mean"]]
    top_neg = df.nsmallest(5, "sentiment_mean")[["post_date", "post_count", "sentiment_mean"]]

    print("\nTop 5 most positive days (sentiment_mean):")
    print(top_pos.to_string(index=False))

    print("\nTop 5 most negative days (sentiment_mean):")
    print(top_neg.to_string(index=False))

    corr_with_volume = (
        df[["post_count", *SENTIMENT_COLS]]
        .corr(numeric_only=True)
        .loc["post_count", SENTIMENT_COLS]
        .round(4)
    )
    print("\nCorrelation with post_count:")
    print(corr_with_volume)

    rolling = df.sort_values("post_date").set_index("post_date")["sentiment_mean"].rolling(7).mean()
    if not rolling.dropna().empty:
        latest = rolling.dropna().tail(5)
        print("\nLatest 5 values of 7-day rolling sentiment_mean:")
        print(latest.round(4).to_string())

    print("\n" + "=" * 70)
    print("Word-level sentiment drivers")
    print("=" * 70)
    print_top_sentiment_words(df, posts_df, top_days=top_days, top_words=top_words)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect NLP sentiment features from trump_daily_features.csv"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to dataset (default: ../data/trump_daily_features.csv)",
    )
    parser.add_argument(
        "--posts-csv",
        type=Path,
        default=DEFAULT_POSTS,
        help="Path to post-level dataset (default: ../data/trump_clean_posts.csv)",
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
        "--plot-last-months",
        type=int,
        default=2,
        help="How many months to include in the sentiment plot (default: 2).",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path(__file__).resolve().parent / "sentiment_last_2_months.png",
        help="Path for saved sentiment plot PNG.",
    )
    args = parser.parse_args()

    df = load_dataset(args.csv)
    posts_df = load_posts(args.posts_csv)
    print_report(df, posts_df, top_days=args.top_days, top_words=args.top_words)
    save_sentiment_plot(df, months=args.plot_last_months, output_path=args.plot_output)


if __name__ == "__main__":
    main()
