import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
import seaborn as sns

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.multioutput import MultiOutputRegressor

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import PartialDependenceDisplay

from sklearn.model_selection import ParameterGrid
import shap
from matplotlib.ticker import ScalarFormatter

from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np


def evaluate_model(name, model, X_train, Y_train, X_val, Y_val):
    model.fit(X_train, Y_train)

    preds = model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(Y_val, preds))
    mae = mean_absolute_error(Y_val, preds)
    r2 = r2_score(Y_val, preds)

    return {
        "model_name": name,
        "model": model,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }


def run_all_models(models, X_train, Y_train, X_val, Y_val):
    results = []

    for name, model in models.items():
        result = evaluate_model(
            name,
            model,
            X_train,
            Y_train,
            X_val,
            Y_val
        )

        results.append(result)

    results_df = pd.DataFrame(results).drop(columns=["model"])
    results_df = results_df.sort_values("rmse")

    return results, results_df


def diagnose_random_forest(
    model,
    X,
    Y,
    time_data=None,
    target_idx=0,
    alpha=None,
    lags=40,
    top_n=15,
    title=None
):
    """
    Diagnostic plots for RandomForest / MultiOutputRegressor(RandomForestRegressor).

    Produces:
    1. ACF of residuals
    2. QQ plot of residuals
    3. Feature importances
    4. Actual vs predicted over time
    """

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------
    y_pred = model.predict(X)

    if y_pred.ndim == 2:
        y_pred_target = y_pred[:, target_idx]
    else:
        y_pred_target = y_pred

    if isinstance(Y, pd.DataFrame):
        target_name = Y.columns[target_idx]
        y_true = Y.iloc[:, target_idx].values
    else:
        target_name = f"target_{target_idx}"
        y_true = np.asarray(Y)
        if y_true.ndim == 2:
            y_true = y_true[:, target_idx]

    residuals = y_true - y_pred_target

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_target))
    mae = mean_absolute_error(y_true, y_pred_target)
    r2 = r2_score(y_true, y_pred_target)

    print(f"Target: {target_name}")
    print(f"RMSE:   {rmse:.6f}")
    print(f"MAE:    {mae:.6f}")
    print(f"R2:     {r2:.6f}")

    # --------------------------------------------------------
    # Extract feature importances
    # --------------------------------------------------------
    if hasattr(model, "estimators_"):
        # MultiOutputRegressor
        rf = model.estimators_[target_idx]
    else:
        # Plain RandomForestRegressor
        rf = model

    importances = pd.Series(
        rf.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    top_importances = importances.head(top_n).sort_values()

    # --------------------------------------------------------
    # Time handling
    # --------------------------------------------------------
    if time_data is None:
        time_data = pd.Series(X.index, index=X.index)
    else:
        time_data = pd.Series(time_data, index=X.index)

    time_data = pd.to_datetime(time_data, errors="coerce")

    plot_df = pd.DataFrame({
        "time": time_data,
        "actual": y_true,
        "predicted": y_pred_target,
        "residual": residuals
    }, index=X.index).dropna()

    # --------------------------------------------------------
    # Plotting
    # --------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    # 1. ACF of residuals
    plot_acf(plot_df["residual"], lags=min(lags, len(plot_df) - 1), ax=axes[0])
    axes[0].set_title("ACF of Residuals")

    # 2. QQ plot
    sm.qqplot(plot_df["residual"], line="s", ax=axes[1])
    axes[1].set_title("QQ Plot of Residuals")

    # 3. Feature importances
    axes[2].barh(top_importances.index, top_importances.values)
    axes[2].set_title(f"Top {top_n} Feature Importances")
    axes[2].set_xlabel("Importance")

    # 4. Actual vs predicted over time
    axes[3].scatter(
        plot_df["time"],
        plot_df["actual"],
        alpha=0.5,
        label="Actual"
    )

    axes[3].plot(
        plot_df["time"],
        plot_df["predicted"],
        linewidth=2,
        label="Predicted"
    )

    axes[3].set_xlim(plot_df["time"].min(), plot_df["time"].max())

    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    formatter = mdates.DateFormatter("%Y-%m-%d")

    axes[3].xaxis.set_major_locator(locator)
    axes[3].xaxis.set_major_formatter(formatter)

    axes[3].set_title("Actual vs Predicted Over Time")
    axes[3].set_xlabel("Date")
    axes[3].set_ylabel(target_name)
    axes[3].legend()
    axes[3].tick_params(axis="x", rotation=45)

    if title is not None:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return {
        "target": target_name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "feature_importances": importances,
        "residuals": plot_df["residual"],
        "predictions": plot_df["predicted"],
        "actual": plot_df["actual"]
    }

def shap_analysis_random_forest(
    model,
    X,
    target_idx=0,
    target_name=None,
    sample_size=500
):
    """
    Compute and display SHAP summary plot for a given target
    from a MultiOutput RandomForest model.
    """

    # Select a model and sample data
    if hasattr(model, "estimators_"):
        rf = model.estimators_[target_idx]
    else:
        rf = model
    if target_name is None:
        target_name = f"target_{target_idx}"
    if len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=42)
    else:
        X_sample = X.copy()

    # Compute shap
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_sample)

    # Plot
    print("=" * 80)
    print(f"SHAP Summary — {target_name}")
    print("=" * 80)

    shap.summary_plot(
        shap_values,
        X_sample,
        plot_size=(12, 6),
        show=True
    )

    return {
        "target": target_name,
        "shap_values": shap_values,
        "data": X_sample
    }

def shap_analysis_side_by_side(model, X, target_names, sample_size=500):
    """
    Plot SHAP summary plots for all targets side-by-side.
    """

    n_targets = len(target_names)

    # Sample data
    if len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=42)
    else:
        X_sample = X.copy()

    fig, axes = plt.subplots(1, n_targets, figsize=(6 * n_targets, 5))

    if n_targets == 1:
        axes = [axes]

    for i, (ax, target_name) in enumerate(zip(axes, target_names)):

        # Get model for target
        if hasattr(model, "estimators_"):
            rf = model.estimators_[i]
        else:
            rf = model

        # Compute SHAP
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_sample)

        # Plot on specific axis
        shap.summary_plot(
            shap_values,
            X_sample,
            show=False,
            plot_size=None  
        )

        # Move plot to correct axis
        plt.sca(ax)
        ax.set_title(target_name)

    plt.tight_layout()
    plt.show()

def pdp_analysis_side_by_side(
    model,
    X,
    target_names,
    features_to_plot=None,
    top_n=5,
    sample_size=500
):
    """
    Plot PDPs for all targets side-by-side.

    If features_to_plot is None, the top_n most important features
    are selected separately for each target.
    """

    n_targets = len(target_names)

    # Sample data
    if len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=42)
    else:
        X_sample = X.copy()

    fig, axes = plt.subplots(
        1,
        n_targets,
        figsize=(7 * n_targets, 5),
        squeeze=False
    )

    axes = axes.ravel()

    for i, (ax, target_name) in enumerate(zip(axes, target_names)):

        # Get model for target
        if hasattr(model, "estimators_"):
            rf = model.estimators_[i]
        else:
            rf = model

        # Choose features
        if features_to_plot is None:
            importances = pd.Series(
                rf.feature_importances_,
                index=X_sample.columns
            ).sort_values(ascending=False)

            selected_features = importances.head(top_n).index.tolist()
        else:
            selected_features = features_to_plot

        # PDP
        PartialDependenceDisplay.from_estimator(
            rf,
            X_sample,
            features=selected_features,
            ax=ax,
            kind="average"
        )

        ax.set_title(f"PDP — {target_name}")

    
    plt.show()

def plot_rf_grid_results(grid_results_df, best_params, metric="r2", top_n=20):
    df_plot = grid_results_df.copy()

    # Create readable config string
    df_plot["config"] = (
        "depth=" + df_plot["max_depth"].astype(str) +
        ", leaf=" + df_plot["min_samples_leaf"].astype(str) +
        ", split=" + df_plot["min_samples_split"].astype(str) +
        ", feat=" + df_plot["max_features"].astype(str) +
        ", trees=" + df_plot["n_estimators"].astype(str)
    )

    # Sort depending on metric
    ascending = metric in ["rmse", "mae", "mse"]
    df_plot = df_plot.sort_values(metric, ascending=ascending)

    df_top = df_plot.head(top_n)

    # Identify the row corresponding to best_params
    best_mask = (
        (df_top["max_depth"] == best_params["max_depth"]) &
        (df_top["max_features"] == best_params["max_features"]) &
        (df_top["min_samples_leaf"] == best_params["min_samples_leaf"]) &
        (df_top["min_samples_split"] == best_params["min_samples_split"]) &
        (df_top["n_estimators"] == best_params["n_estimators"])
    )

    # Default colors, highlight best model in red
    colors = ["red" if is_best else "steelblue" for is_best in best_mask]

    plt.figure(figsize=(14, 7))
    plt.barh(
        df_top["config"],
        df_top[metric],
        color=colors
    )

    plt.xlabel(metric.upper())
    plt.xlim(df_top[metric].min() * 0.9, df_top[metric].max() * 1.01)
    plt.ylabel("Parameter configuration")
    plt.title(f"Top {top_n} Random Forest Configurations by {metric.upper()}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def encode_for_parallel_plot(df):
    df = df.copy()

    for col in ["max_depth", "max_features"]:
        df[col] = df[col].astype(str)
        mapping = {v: i for i, v in enumerate(sorted(df[col].unique()))}
        df[f"{col}_encoded"] = df[col].map(mapping)

    return df


def diagnose_predictions(y_true, y_pred, target_names=None):
    y_true = pd.DataFrame(y_true)
    y_pred = pd.DataFrame(y_pred)

    if target_names is None:
        target_names = y_true.columns

    for i, target in enumerate(target_names):
        yt = y_true.iloc[:, i].values
        yp = y_pred.iloc[:, i].values
        residuals = yt - yp

        fig, ax = plt.subplots(1, 4, figsize=(22, 6))

        ax[0].scatter(yp, residuals, alpha=0.6)
        ax[0].axhline(0, linestyle="--")
        ax[0].set_title(f"{target}: Residuals vs Predicted")
        ax[0].set_xlabel("Predicted")
        ax[0].set_ylabel("Residuals")

        ax[1].hist(residuals, bins=30)
        ax[1].set_title(f"{target}: Residual Distribution")
        ax[1].set_xlabel("Residuals")

        stats.probplot(residuals, dist="norm", plot=ax[2])
        ax[2].set_title(f"{target}: Q-Q Plot")

        
        ax[3].plot(yp, label="Predicted")
        ax[3].scatter(np.arange(len(yt)), yt, alpha=0.5, label="Actual")
        ax[3].set_title(f"{target}: Actual vs Predicted\nR² = {r2_score(yt, yp):.3f}")
        ax[3].set_xlabel("Sample Index")
        ax[3].set_ylabel(target)
        ax[3].legend()

        plt.tight_layout()
        plt.show()



def plot_actual_vs_predicted(
    model,
    X_train, Y_train,
    X_val, Y_val,
    X_test, Y_test,
    target_name
):
    # Get column index
    target_idx = Y_train.columns.get_loc(target_name)

    # Predict only selected target
    train_pred = model.predict(X_train)[:, target_idx]
    val_pred   = model.predict(X_val)[:, target_idx]
    test_pred  = model.predict(X_test)[:, target_idx]

    # Actuals for selected target
    y_train = Y_train[target_name].values
    y_val   = Y_val[target_name].values
    y_test  = Y_test[target_name].values

    print(f"{target_name} Train R²:", r2_score(y_train, train_pred))
    print(f"{target_name} Val R²:", r2_score(y_val, val_pred))
    print(f"{target_name} Test R²:", r2_score(y_test, test_pred))

    y_all = np.concatenate([y_train, y_val, y_test])

    train_idx = np.arange(len(y_train))
    val_idx   = np.arange(len(y_train), len(y_train) + len(y_val))
    test_idx  = np.arange(len(y_train) + len(y_val), len(y_all))

    fig, ax = plt.subplots(figsize=(18, 6))

    # Actual values
    ax.scatter(np.arange(len(y_all)), y_all, alpha=0.5, label="Actual")

    # Predictions
    ax.plot(train_idx, train_pred, color="blue", linewidth=2, label="Train Prediction")
    ax.plot(val_idx, val_pred, color="gold", linewidth=2, label="Validation Prediction")
    ax.plot(test_idx, test_pred, color="red", linewidth=2, label="Test Prediction")

    # Split markers
    ax.axvline(len(y_train), linestyle="--", alpha=0.7)
    ax.axvline(len(y_train) + len(y_val), linestyle="--", alpha=0.7)

    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(style="plain", axis="y")
    ax.set_ylim(0, max(y_all.max(), train_pred.max(), val_pred.max(), test_pred.max()) * 1.1)

    ax.set_xlabel("Sample Index")
    ax.set_ylabel(target_name)
    ax.set_title(f"Actual vs Predicted — {target_name}")
    ax.legend()

    plt.tight_layout()
    plt.show()

def shap_local_waterfall(model, X, target_names, sample_indices=None):
    """
    Plot SHAP waterfall plots for individual observations (local explainability).

    Shows how each feature pushes a specific prediction above or below the
    base (mean training) value — one plot per observation per target.
    """
    if sample_indices is None:
        n = len(X)
        sample_indices = [0, n // 2, n - 1]

    for i, target_name in enumerate(target_names):
        if hasattr(model, "estimators_"):
            rf = model.estimators_[i]
        else:
            rf = model

        explainer = shap.TreeExplainer(rf)
        shap_vals = explainer(X)

        print("=" * 70)
        print(f"Local SHAP Waterfall — {target_name}")
        print("=" * 70)

        for idx in sample_indices:
            pred = rf.predict(X.iloc[[idx]])[0]
            print(f"\nSample {idx}  |  Predicted {target_name}: {pred:,.0f}")
            shap.plots.waterfall(shap_vals[idx], max_display=10, show=True)


def plot_actual_vs_predicted_with_sentiment(
    model,
    X_val, Y_val,
    X_test, Y_test,
    val, test,
    target_name
):
    target_idx = Y_val.columns.get_loc(target_name)

    val_pred  = model.predict(X_val)[:, target_idx]
    test_pred = model.predict(X_test)[:, target_idx]

    y_val  = Y_val[target_name].values
    y_test = Y_test[target_name].values

    print(f"{target_name} Val R²:", r2_score(y_val, val_pred))
    print(f"{target_name} Test R²:", r2_score(y_test, test_pred))

    y_all = np.concatenate([y_val, y_test])
    pred_all = np.concatenate([val_pred, test_pred])

    df_plot = pd.concat([val, test], axis=0).reset_index(drop=True)

    x = np.arange(len(y_all))

    val_idx  = np.arange(len(y_val))
    test_idx = np.arange(len(y_val), len(y_all))

    fig, ax = plt.subplots(figsize=(18, 6))

    # Actual values
    ax.scatter(x, y_all, alpha=0.5, label="Actual")

    # Predictions
    ax.plot(val_idx, val_pred, color="gold", linewidth=2, label="Validation Prediction")
    ax.plot(test_idx, test_pred, color="red", linewidth=2, label="Test Prediction")

    # Split marker between val and test
    ax.axvline(len(y_val), linestyle="--", alpha=0.7)

    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(style="plain", axis="y")
    ax.set_ylim(0, max(y_all.max(), pred_all.max()) * 1.1)

    ax.set_xlabel("Sample Index")
    ax.set_ylabel(target_name)
    ax.set_title(f"Actual vs Predicted — {target_name} (Validation + Test)")

    # Secondary axis for sentiment
    ax2 = ax.twinx()
    ax2.plot(
        x,
        df_plot["trump_sentiment_pct_negative"].values,
        color="purple",
        linestyle=":",
        linewidth=2,
        label="Trump Negative Sentiment %"
    )
    ax2.set_ylabel("Trump Negative Sentiment %")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.show()

def plot_actual_vs_predicted_with_topics(
    model,
    X_val, Y_val,
    X_test, Y_test,
    val, test,
    target_name
):
    target_idx = Y_val.columns.get_loc(target_name)

    val_pred  = model.predict(X_val)[:, target_idx]
    test_pred = model.predict(X_test)[:, target_idx]

    y_val  = Y_val[target_name].values
    y_test = Y_test[target_name].values

    print(f"{target_name} Val R²:", r2_score(y_val, val_pred))
    print(f"{target_name} Test R²:", r2_score(y_test, test_pred))

    y_all = np.concatenate([y_val, y_test])
    pred_all = np.concatenate([val_pred, test_pred])
    df_plot = pd.concat([val, test], axis=0).reset_index(drop=True)

    x = np.arange(len(y_all))
    val_idx  = np.arange(len(y_val))
    test_idx = np.arange(len(y_val), len(y_all))

    fig, ax = plt.subplots(figsize=(18, 6))

    ax.scatter(x, y_all, alpha=0.5, label="Actual")
    ax.plot(val_idx, val_pred, color="gold", linewidth=2, label="Validation Prediction")
    ax.plot(test_idx, test_pred, color="red", linewidth=2, label="Test Prediction")

    # Mark all topics 0–12 with black circles and text labels
    for topic_id in range(13):
        col = f"topic_activity_{topic_id}"
        topic_mask = df_plot[col].values == 1

        ax.scatter(
            x[topic_mask],
            y_all[topic_mask],
            s=90,
            facecolors="none",
            edgecolors="black",
            linewidths=1.5
        )

        for xi, yi in zip(x[topic_mask], y_all[topic_mask]):
            ax.annotate(
                f"Topic {topic_id}",
                xy=(xi, yi),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color="black"
            )

    ax.axvline(len(y_val), linestyle="--", alpha=0.7)

    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(style="plain", axis="y")
    ax.set_ylim(0, max(y_all.max(), pred_all.max()) * 1.1)

    ax.set_xlabel("Sample Index")
    ax.set_ylabel(target_name)
    ax.set_title(f"Actual vs Predicted — {target_name} (Validation + Test)")



    lines1, labels1 = ax.get_legend_handles_labels()
    ax.legend(lines1 , labels1, loc="upper left")

    plt.tight_layout()
    plt.show()