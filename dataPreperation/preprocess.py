import pandas as pd
import nltk
import datetime
from nltk.sentiment import SentimentIntensityAnalyzer
import yfinance as yf
nltk.download("vader_lexicon")

########### ---------- Helper functions - Trump data ---------- ###########
'''
    Simple loader, where er pick the columns that we want to use
'''
def load_tweets(file_path):
    df = pd.read_json(file_path, lines=True)
    df = df[['created_at', 'content', 'upvotes_count']]
    print(f"Loaded {len(df)} tweets from {file_path}")
    return df


'''
    Ensure we have a datetime time column
'''
def _add_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df = df.dropna(subset=['created_at'])
    df['date'] = df['created_at'].dt.normalize()
    return df


'''
    Just added a sentiment score as a proof of concept, we should probly make something more cool
'''
def _add_sentiment_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sia = SentimentIntensityAnalyzer()
    df['sentiment'] = df['content'].fillna('').apply(
        lambda text: sia.polarity_scores(text)['compound']
    )
    return df


'''
    Thought that some keywords might be interesting to track.
'''
def _add_keyword_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text = df['content'].fillna('').str.lower()

    df['mention_israel'] = text.str.contains(r'\bisrael\b|\bisreal\b', regex=True).astype(int)
    df['mention_greenland'] = text.str.contains(r'\bgreenland\b', regex=True).astype(int)

    return df


'''
    Apply features per tweet, meaning that we still have multiple lines per day
'''
def apply_tweet_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add tweet-level features.
    Easy to extend by adding more `_add_*_feature` functions here.
    """
    df = _add_date(df)
    df = _add_sentiment_feature(df)
    df = _add_keyword_features(df)
    return df

'''
    To match stock data we group the tweets per day.
    Here we decide the aggregation method
'''
def aggregate_tweet_features_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate tweet-level features to one row per day.
    Change mean/sum/count here depending on how you want each feature treated.
    """
    daily_df = (
        df.groupby('date')
        .agg(
            tweet_count=('content', 'count'),
            avg_upvotes=('upvotes_count', 'mean'),
            avg_sentiment=('sentiment', 'mean'),
            mention_israel=('mention_israel', 'max'),
            mention_greenland=('mention_greenland', 'max'),
        )
        .reset_index()
    )

    return daily_df



########### ---------- Helper function - stock data ---------- ###########
"""
Args:
    ticker (str): Ticker symbol (e.g., 'SPY', '^GSPC')
    n_months (int): Number of months of historical data
"""
def get_last_n_months_data(ticker: str, n_months: int) -> pd.DataFrame:
    # Download last n month
    end_date = datetime.datetime.today()
    start_date = end_date - pd.DateOffset(months=n_months)

    data = yf.download(ticker, start=start_date, end=end_date)['Close']
    data = data.reset_index()
    data.columns = ['date', 'price']

    return data

########### ---------- Putting it all together ---------- ###########

def get_time_series_data(trump_data_path, ticker, n_months):
    # Load and process Trump tweets
    trump_df = load_tweets(trump_data_path)
    tweet_feature_df = apply_tweet_features(trump_df)
    daily_tweet_df = aggregate_tweet_features_daily(tweet_feature_df)

    # Load stock data
    stock_df = get_last_n_months_data(ticker, n_months)

    # Had some timezone error
    daily_tweet_df['date'] = daily_tweet_df['date'].dt.tz_localize(None)
    stock_df['date'] = stock_df['date'].dt.tz_localize(None)

    merged_df = pd.merge(daily_tweet_df, stock_df, on='date', how='inner')

    return merged_df
