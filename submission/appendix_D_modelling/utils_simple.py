import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.preprocessing import PolynomialFeatures


def train_model_ols(df, features, target, intercept_category='topic_activity_0'):
    
    # Define variables
    X = df[features].drop(columns=[intercept_category])
    y = df[target]  

    # add intercept and fit
    X = sm.add_constant(X)
    model = sm.OLS(y, X, missing='drop').fit()

    return model

def train_model_ols_second_order(df, features, target, intercept_category='topic_activity_0'):
    # Define variables
    X = df[features].drop(columns=[intercept_category], errors='ignore')
    y = df[target]

    # Add second-order terms: squares + pairwise interactions
    poly = PolynomialFeatures(
        degree=2,
        include_bias=False
    )

    X_poly = poly.fit_transform(X)
    X_poly = pd.DataFrame(
        X_poly,
        columns=poly.get_feature_names_out(X.columns),
        index=X.index
    )

    # Add intercept and fit
    X_poly = sm.add_constant(X_poly)
    model = sm.OLS(y, X_poly, missing='drop').fit()

    return model

def train_model_arx(df, features, target, y_lags=1, x_lags=1):

    base = df.copy()

    # --- Target lags ---
    y_lag_df = pd.concat(
        {
            f"{target}_lag_{lag}": base[target].shift(lag)
            for lag in range(1, y_lags + 1)
        },
        axis=1
    )

    # --- Feature lags ---
    x_lag_dict = {}

    for feature in features:
        if feature == "topic_activity_0":
            continue

        # current value
        x_lag_dict[feature] = base[feature]

        # lagged values
        for lag in range(1, x_lags + 1):
            x_lag_dict[f"{feature}_lag_{lag}"] = base[feature].shift(lag)

    x_lag_df = pd.DataFrame(x_lag_dict)

    # --- Combine everything at once ---
    X = pd.concat([y_lag_df, x_lag_df], axis=1)
    y = base[target]

    X = sm.add_constant(X)

    model = sm.OLS(y, X, missing="drop").fit()

    return model

def diagnose_model(model, time_data, alpha=0.05, lags=40, title=None):

    residuals = model.resid

    # --- feature stats ---
    coef = model.params.drop("const", errors="ignore")
    pvals = model.pvalues.drop("const", errors="ignore")
    significant = pvals < alpha

    # sort by absolute magnitude
    order = coef.abs().sort_values().index
    coef_sorted = coef.loc[order]
    significant_sorted = significant.loc[order]

    # --- predictions ---
    y_true = model.model.endog
    y_pred = model.fittedvalues

    # --- plotting ---
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    # 1. ACF
    plot_acf(residuals, lags=lags, ax=axes[0])
    axes[0].set_title("ACF of Residuals")

    # 2. QQ plot
    sm.qqplot(residuals, line='s', ax=axes[1])
    axes[1].set_title("QQ Plot of Residuals")

    # 3. Feature contributions
    top_n = 15
    coef_top = coef_sorted.reindex(coef_sorted.abs().sort_values(ascending=False).head(top_n).index)
    significant_top = significant_sorted.loc[coef_top.index]

    axes[2].barh(coef_top.index, coef_top.values)

    for i, (feature, value) in enumerate(coef_top.items()):
        if significant_top.loc[feature]:
            axes[2].text(
                value,
                i,
                "  *",
                va="center",
                fontsize=12,
                fontweight="bold"
            )

    axes[2].axvline(0, linewidth=1)
    axes[2].set_title(f"Top {top_n} Feature Contributions (p < {alpha})")
    axes[2].set_xlabel("Coefficient")

    # 4. Actual vs Predicted over time
    y_true = pd.Series(model.model.endog)
    y_pred = pd.Series(model.fittedvalues)

    # force real datetimes
    time_data = pd.to_datetime(time_data, errors="coerce")

    # align lengths after OLS dropped missing rows
    valid_idx = model.model.data.row_labels
    time_data = time_data.loc[valid_idx]

    # remove any bad dates
    mask = time_data.notna()
    time_data = time_data[mask]
    y_true = y_true[mask.to_numpy()]
    y_pred = y_pred[mask.to_numpy()]

    axes[3].scatter(time_data, y_true, alpha=0.5, label="Actual")
    axes[3].plot(time_data, y_pred, linewidth=2, label="Predicted")

    axes[3].set_xlim(time_data.min(), time_data.max())

    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    formatter = mdates.DateFormatter("%Y-%m-%d")

    axes[3].xaxis.set_major_locator(locator)
    axes[3].xaxis.set_major_formatter(formatter)

    axes[3].set_title("Actual vs Predicted (Time)")
    axes[3].set_xlabel("Date")
    axes[3].set_ylabel("Value")
    axes[3].legend()
    axes[3].tick_params(axis="x", rotation=45)

    # --- global title ---
    if title is not None:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def diagnose_validation(model, val_df, features, target, time_col="date",
                        intercept_category="topic_activity_0",
                        alpha=0.05, lags=40, title=None):

    # --- rebuild validation X like train_model_ols ---
    X_val = val_df[features].drop(columns=[intercept_category], errors="ignore")
    y_val = val_df[target]

    X_val = sm.add_constant(X_val, has_constant="add")

    # align validation columns to training model columns
    X_val = X_val.reindex(columns=model.model.exog_names, fill_value=0)

    # drop rows with missing y or X
    valid = pd.concat([y_val, X_val], axis=1).dropna().index
    y_true = y_val.loc[valid]
    X_val = X_val.loc[valid]

    y_pred = model.predict(X_val)
    residuals = y_true - y_pred

    time_data = pd.to_datetime(val_df.loc[valid, time_col], errors="coerce")

    # --- plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    plot_acf(residuals, lags=min(lags, len(residuals) - 1), ax=axes[0])
    axes[0].set_title("Validation Residual ACF")

    sm.qqplot(residuals, line="s", ax=axes[1])
    axes[1].set_title("Validation Residual QQ Plot")

    mask = time_data.notna()

    axes[2].scatter(time_data[mask], y_true[mask], alpha=0.5, label="Actual")
    axes[2].plot(time_data[mask], y_pred[mask], linewidth=2, label="Predicted")

    axes[2].set_title("Validation: Actual vs Predicted")
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Value")
    axes[2].legend()
    axes[2].tick_params(axis="x", rotation=45)

    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    formatter = mdates.DateFormatter("%Y-%m-%d")
    axes[2].xaxis.set_major_locator(locator)
    axes[2].xaxis.set_major_formatter(formatter)

    if title is not None:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return {
        "mse": np.mean(residuals ** 2),
        "rmse": np.sqrt(np.mean(residuals ** 2)),
        "mae": np.mean(np.abs(residuals)),
        "r2_validation": 1 - np.sum(residuals ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    }