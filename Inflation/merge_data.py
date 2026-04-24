import pandas as pd


def clean_fred(file_path, value_name):
    df = pd.read_csv(file_path)
    df = df.rename(columns={df.columns[0]: "date", df.columns[1]: value_name})
    df["date"] = pd.to_datetime(df["date"])
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    df = df.sort_values("date").dropna()
    return df


def load_macro_data(base_path):
    cpi = clean_fred(f"{base_path}/CPIAUCSL.csv", "cpi")
    fed = clean_fred(f"{base_path}/FEDFUNDS.csv", "interest_rate")
    sent = clean_fred(f"{base_path}/UMCSENT.csv", "consumer_sentiment")
    unemp = clean_fred(f"{base_path}/UNRATE.csv", "unemployment")
    return cpi, fed, sent, unemp


def merge_macro_data(cpi, fed, sent, unemp):
    df = (
        cpi.merge(fed, on="date", how="inner")
           .merge(sent, on="date", how="inner")
           .merge(unemp, on="date", how="inner")
           .sort_values("date")
    )
    return df


def create_daily_frame(df, start_date, end_date, buffer_start):
    daily_dates = pd.date_range(start=buffer_start, end=end_date, freq="D")
    daily_df = pd.DataFrame({"date": daily_dates})
    df_daily = daily_df.merge(df, on="date", how="left").sort_values("date").ffill()
    df_daily = df_daily[
        (df_daily["date"] >= start_date) &
        (df_daily["date"] <= end_date)
    ]
    return df_daily


def save_output(df, output_path):
    df.to_csv(output_path, index=False)


def main():
    base_path = "inflation/FRED_data"
    output_path = "inflation/merged_macro_daily_2025_2026.csv"

    start_date = "2025-11-14"
    end_date = "2026-04-10"
    buffer_start = "2025-01-01"

    cpi, fed, sent, unemp = load_macro_data(base_path)
    df = merge_macro_data(cpi, fed, sent, unemp)
    df_daily = create_daily_frame(df, start_date, end_date, buffer_start)

    save_output(df_daily, output_path)

    print("Saved to:", output_path)
    print(df_daily.head())
    print(df_daily.shape)


if __name__ == "__main__":
    main()