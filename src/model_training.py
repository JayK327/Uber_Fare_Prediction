import os, json, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

PLOT_DIR = "outputs/plots"
MODEL_DIR = "models"
TARGET = "fare_amount"
RS = 42

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("outputs", exist_ok=True)

def metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-6, None))) * 100
    return dict(rmse=round(rmse,4), mae=round(mae,4), r2=round(r2,4), mape=round(mape,3))

def time_split(df, feature_cols, test_year=2015):
    tr = df["year"] < test_year
    return (
        df.loc[tr, feature_cols],
        df.loc[~tr, feature_cols],
        df.loc[tr, TARGET],
        df.loc[~tr, TARGET]
    )

def build_models():
    return {
        "LinearRegression": Pipeline([("sc", StandardScaler()), ("m", LinearRegression())]),
        "Ridge": Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=10.0))]),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=12,
            min_samples_leaf=4, n_jobs=-1, random_state=RS
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=7,
            subsample=0.8, colsample_bytree=0.8,
            objective="reg:squarederror",
            n_jobs=-1, random_state=RS, verbosity=0
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05,
            num_leaves=63, subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1, random_state=RS, verbose=-1
        ),
    }

def train_all(X_tr, X_te, y_tr, y_te):
    results = {}

    print("\nModel                  Train RMSE   Test RMSE   Test R²")
    print("-" * 60)

    for name, model in build_models().items():
        model.fit(X_tr, y_tr)

        tr_pred = model.predict(X_tr)
        te_pred = model.predict(X_te)

        tr_m = metrics(y_tr, tr_pred)
        te_m = metrics(y_te, te_pred)

        results[name] = {"train": tr_m, "test": te_m}

        print(f"{name:<22} {tr_m['rmse']:>10} {te_m['rmse']:>10} {te_m['r2']:>10}")

        joblib.dump(model, f"{MODEL_DIR}/{name.lower()}.joblib")

    return results

def tune_lgbm(X_tr, y_tr):
    print("[DEBUG] entered tuning function")

    search = RandomizedSearchCV(
        lgb.LGBMRegressor(random_state=RS, verbose=-1),
        {
            "n_estimators": [300,500],
            "learning_rate": [0.03,0.05],
            "num_leaves": [31,63],
            "max_depth": [-1,6],
        },
        n_iter=4,
        cv=2,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=0
    )

    search.fit(X_tr, y_tr)

    print(f"[HPO] Best RMSE: {-search.best_score_:.4f}")
    print(f"[HPO] Params: {search.best_params_}")

    return search.best_estimator_

def plot_feature_importance(model, cols):
    imp = model.feature_importances_
    fi = pd.Series(imp, index=cols).sort_values(ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(9,6))
    ax.barh(fi.index[::-1], fi.values[::-1], color="#2563eb")
    ax.set_title("Feature Importance")

    path = f"{PLOT_DIR}/feature_importance.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("[SAVED]", path)

def plot_predictions(y_true, y_pred):
    fig, ax = plt.subplots(1,2, figsize=(12,5))

    ax[0].scatter(y_true, y_pred, s=5, alpha=0.3)
    ax[0].set_title("Actual vs Predicted")

    ax[1].hist(y_true - y_pred, bins=50, color="orange")
    ax[1].set_title("Residuals")

    path = f"{PLOT_DIR}/predictions.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("[SAVED]", path)

def run(df, feature_cols, tune=True):

    print("Shape:", df.shape)
    missing = [c for c in feature_cols if c not in df.columns]
    print("Missing features:", missing)

    X_tr, X_te, y_tr, y_te = time_split(df, feature_cols)

    results = train_all(X_tr, X_te, y_tr, y_te)

    lgbm = joblib.load(f"{MODEL_DIR}/lightgbm.joblib")

    plot_feature_importance(lgbm, feature_cols)
    plot_predictions(y_te, lgbm.predict(X_te))

    if tune:
        print("[DEBUG] about to call tuning")

        tuned = tune_lgbm(X_tr, y_tr)
        joblib.dump(tuned, f"{MODEL_DIR}/lgbm_tuned.joblib")

        print("[DEBUG] tuned model saved")

        results["LightGBM_Tuned"] = {
            "train": metrics(y_tr, tuned.predict(X_tr)),
            "test": metrics(y_te, tuned.predict(X_te)),
        }

    with open("outputs/model_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nFINAL LEADERBOARD")
    print("-" * 60)
    print(f"{'Model':<22}{'RMSE':>10}{'R²':>10}")
    print("-" * 60)

    for k, v in sorted(results.items(), key=lambda x: x[1]["test"]["rmse"]):
        print(f"{k:<22}{v['test']['rmse']:>10}{v['test']['r2']:>10}")

    print("\nDONE ")

if __name__ == "__main__":
    df = pd.read_csv("data/processed/uber_features.csv")
    from src.feature_engineering import FEATURE_COLS
    run(df, FEATURE_COLS, tune=True)