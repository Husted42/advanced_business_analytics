import pandas as pd

def clean_fred(file_path, value_name):
    df = pd.read_csv(file_path) 
    
    df = df.rename(columns={
        df.columns[0]: "date",
        df.columns[1]: value_name
    })
    
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df = df.dropna()
    
    return df

# Clean each dataset
cpi = clean_fred("/Users/michellemai/Documents/GitHub/advanced_business_analytics/inflation/FRED_data/CPIAUCSL.csv", "cpi")
fed = clean_fred("/Users/michellemai/Documents/GitHub/advanced_business_analytics/inflation/FRED_data/FEDFUNDS.csv", "interest_rate")
sent = clean_fred("/Users/michellemai/Documents/GitHub/advanced_business_analytics/inflation/FRED_data/UMCSENT.csv", "consumer_sentiment")
unemp = clean_fred("/Users/michellemai/Documents/GitHub/advanced_business_analytics/inflation/FRED_data/UNRATE.csv", "unemployment")


df = cpi.merge(fed, on="date", how="inner") \
        .merge(sent, on="date", how="inner") \
        .merge(unemp, on="date", how="inner")

start_date = "2016-01-01"
end_date = "2021-01-20"

df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
df = df.sort_values("date") 

# Create features
df["inflation"] = df["cpi"].pct_change() * 100
df["interest_rate_change"] = df["interest_rate"].diff()
df["unemployment_change"] = df["unemployment"].diff()
df["sentiment_change"] = df["consumer_sentiment"].diff()

# Lag features
for lag in [1, 2, 3]:
    df[f"inflation_lag_{lag}"] = df["inflation"].shift(lag)
    df[f"sentiment_lag_{lag}"] = df["sentiment_change"].shift(lag)

# Drop NA AFTER all transformations
df = df.dropna()

# Define X and y
y = df["sentiment_change"]
X = df.drop(columns=["date", "sentiment_change", "cpi"])

# Scale
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=False  # important for time series
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

import matplotlib.pyplot as plt

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

plt.figure()

plt.plot(df["date"], df["inflation"])

plt.title("Monthly Inflation Rate (%)")
plt.xlabel("Date")
plt.ylabel("Inflation (%)")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure()

plt.plot(df["date"], df["sentiment_change"], label="Sentiment Change")
plt.plot(df["date"], df["inflation"], label="Inflation")

plt.legend()
plt.title("Sentiment vs Inflation Over Time")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

