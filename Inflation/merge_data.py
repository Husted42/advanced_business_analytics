import pandas as pd

def clean_fred(file_path, value_name):
    df = pd.read_csv(file_path)
    
    df = df.rename(columns={
        df.columns[0]: "date",
        df.columns[1]: value_name
    })
    
    df["date"] = pd.to_datetime(df["date"])
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    
    df = df.sort_values("date")
    df = df.dropna()
    
    return df

cpi = clean_fred("/Users/michellemai/Documents/GitHub/advanced_business_analytics/inflation/FRED_data/CPIAUCSL.csv", "cpi")
fed = clean_fred("/Users/michellemai/Documents/GitHub/advanced_business_analytics/inflation/FRED_data/FEDFUNDS.csv", "interest_rate")
sent = clean_fred("/Users/michellemai/Documents/GitHub/advanced_business_analytics/inflation/FRED_data/UMCSENT.csv", "consumer_sentiment")
unemp = clean_fred("/Users/michellemai/Documents/GitHub/advanced_business_analytics/inflation/FRED_data/UNRATE.csv", "unemployment")

df = cpi.merge(fed, on="date", how="inner") \
        .merge(sent, on="date", how="inner") \
        .merge(unemp, on="date", how="inner")

df = df.sort_values("date")

start_date = "2025-11-14"
end_date = "2026-04-10"

daily_dates = pd.date_range(start=start_date, end=end_date, freq="D")
daily_df = pd.DataFrame({"date": daily_dates})

df_daily = daily_df.merge(df, on="date", how="left")

df_daily = df_daily.sort_values("date")
df_daily = df_daily.ffill()

output_path = "merged_macro_daily_2025_2026.csv"
df_daily.to_csv(output_path, index=False)

print("Saved to:", output_path)
print(df_daily.head())
print(df_daily.shape)