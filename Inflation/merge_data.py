import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def clean_fred(file_path, value_name):
    df = pd.read_csv(file_path)
    df = df.rename(columns={df.columns[0]: "date", df.columns[1]: value_name})
    df["date"] = pd.to_datetime(df["date"])
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    return df.sort_values("date").dropna()


def load_macro_data(base_path):
    return {
        "cpi": clean_fred(f"{base_path}/CPIAUCSL.csv", "cpi"),
        "fed": clean_fred(f"{base_path}/FEDFUNDS.csv", "interest_rate"),
        "sentiment": clean_fred(f"{base_path}/UMCSENT.csv", "consumer_sentiment"),
        "unemployment": clean_fred(f"{base_path}/UNRATE.csv", "unemployment"),
    }


def merge_macro_data(data):
    return (
        data["cpi"]
        .merge(data["fed"], on="date", how="inner")
        .merge(data["sentiment"], on="date", how="inner")
        .merge(data["unemployment"], on="date", how="inner")
        .sort_values("date")
    )


def create_daily_macro_data(df, start_date, end_date, buffer_start):
    daily_dates = pd.date_range(start=buffer_start, end=end_date, freq="D")
    daily_df = pd.DataFrame({"date": daily_dates})

    df_daily = daily_df.merge(df, on="date", how="left")
    df_daily = df_daily.sort_values("date").ffill()

    return df_daily[
        (df_daily["date"] >= start_date) &
        (df_daily["date"] <= end_date)
    ]


def create_modeling_data(df, start_date, end_date):
    df = df[
        (df["date"] >= start_date) &
        (df["date"] <= end_date)
    ].copy()

    df = df.sort_values("date")

    df["inflation"] = df["cpi"].pct_change() * 100
    df["interest_rate_change"] = df["interest_rate"].diff()
    df["unemployment_change"] = df["unemployment"].diff()
    df["sentiment_change"] = df["consumer_sentiment"].diff()

    for lag in [1, 2, 3]:
        df[f"inflation_lag_{lag}"] = df["inflation"].shift(lag)
        df[f"sentiment_lag_{lag}"] = df["sentiment_change"].shift(lag)

    return df.dropna()


def create_train_test_split(df):
    y = df["sentiment_change"]
    X = df.drop(columns=["date", "sentiment_change", "cpi"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        shuffle=False,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def plot_macro_indicators(df):
    plt.figure()
    plt.plot(df["date"], df["inflation"], label="Inflation")
    plt.plot(df["date"], df["interest_rate"], label="Interest Rate")
    plt.plot(df["date"], df["unemployment"], label="Unemployment")
    plt.legend()
    plt.title("Macroeconomic Indicators Over Time")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_inflation(df):
    plt.figure()
    plt.plot(df["date"], df["inflation"])
    plt.title("Monthly Inflation Rate (%)")
    plt.xlabel("Date")
    plt.ylabel("Inflation (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_sentiment_vs_inflation(df):
    plt.figure()
    plt.plot(df["date"], df["sentiment_change"], label="Sentiment Change")
    plt.plot(df["date"], df["inflation"], label="Inflation")
    plt.legend()
    plt.title("Sentiment vs Inflation Over Time")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def save_output(df, output_path):
    df.to_csv(output_path, index=False)


def main():
    base_path = "inflation/FRED_data"

    daily_start_date = "2025-10-14"
    daily_end_date = "2026-04-10"
    daily_buffer_start = "2025-01-01"
    daily_output_path = "inflation/merged_macro_daily_2025_2026.csv"

    modeling_start_date = "2016-01-01"
    modeling_end_date = "2021-01-20"

    macro_data = load_macro_data(base_path)
    macro_df = merge_macro_data(macro_data)

    daily_df = create_daily_macro_data(
        macro_df,
        daily_start_date,
        daily_end_date,
        daily_buffer_start,
    )

    save_output(daily_df, daily_output_path)

    modeling_df = create_modeling_data(
        macro_df,
        modeling_start_date,
        modeling_end_date,
    )

    X_train_scaled, X_test_scaled, y_train, y_test, scaler = create_train_test_split(
        modeling_df
    )

    plot_macro_indicators(modeling_df)
    plot_inflation(modeling_df)
    plot_sentiment_vs_inflation(modeling_df)

    print("Saved to:", daily_output_path)
    print(daily_df.head())
    print(daily_df.shape)
    print(X_train_scaled.shape)
    print(X_test_scaled.shape)


if __name__ == "__main__":
    main()