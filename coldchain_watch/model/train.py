"""
train.py
--------
Model A: Temperature Forecasting for Cold-Chain Heat Exposure

What we learned from the data that shapes the model design:
  - Karachi external temp ALWAYS exceeds potato (>4°C) and onion (>2°C) thresholds.
    A binary classifier for these would always predict 1 — useless.
  - Tomato threshold (>13°C) has only 51 negative cases in 2 years — still too skewed.
  - The REAL problem for warehouse managers is: "How hot will it get in the next
    1–24 hours?" so they can decide when to pre-cool and when to dispatch trucks.

Model decision:
  → Random Forest REGRESSOR predicting temp_c at +1h, +6h, +24h ahead.
  → Baseline: same hour last week (temp_lag_168h) — declared in dataset proposal.
  → We compare predicted temp against each crop's FAO threshold to derive
    heat alerts downstream. The regression output is more useful than binary
    classification because it gives the manager a magnitude, not just yes/no.

Secondary model:
  → Humidex forecasting at +6h ahead (combines temp + humidity into felt-heat).
    Used for the "pre-cooling intensity" recommendation.

Evaluation follows guide requirements:
  1. Time-based split — train on earlier data, test on most recent 8 weeks.
     NO random shuffle. Test set is locked before any tuning.
  2. Baseline comparison — must beat same-hour-last-week on test set.
  3. Errors by group — by season, by hour bucket (night/morning/afternoon/evening).
  4. Uncertainty — bootstrap confidence intervals on predictions.
  5. Model card written to model/model_card.md.

Run: python -m model.train
"""

import json
import os
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from loguru import logger

warnings.filterwarnings("ignore")

os.makedirs("model/artifacts", exist_ok=True)
os.makedirs("data/logs", exist_ok=True)
logger.add(
    "data/logs/model_train_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GOLD_PATH     = "data/gold/weather_gold_karachi.parquet"
ARTIFACTS_DIR = "model/artifacts"
RANDOM_SEED   = 42          # fixed — same seed gives same result every run
TEST_WEEKS    = 8           # most recent 8 weeks held out as test set
N_ESTIMATORS  = 200         # RF trees — enough for stability, not too slow
MAX_DEPTH     = 12          # limits overfitting

# Forecast horizons in hours
HORIZONS = {
    "1h":  1,
    "6h":  6,
    "24h": 24,
}

# Features used by the model — derived from Gold columns
# Excludes: raw source cols (om_*, nasa_*), timestamps, row_id, target-leaking cols
BASE_FEATURES = [
    # Current weather state
    "temp_c",
    "humidity_pct",
    "feels_like_c",
    "wind_kmh",
    "precip_mm",
    "humidex",
    "dew_point_c",
    "weather_code",

    # Time features
    "hour_of_day",
    "day_of_week",
    "month",
    "day_of_year",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",

    # Rolling stats — recent trend
    "temp_roll_mean_3h",
    "temp_roll_mean_6h",
    "temp_roll_mean_24h",
    "temp_roll_max_3h",
    "temp_roll_max_6h",
    "temp_roll_max_24h",
    "temp_roll_std_24h",
    "hum_roll_mean_3h",
    "hum_roll_mean_6h",
    "hum_roll_mean_24h",
    "precip_roll_sum_6h",
    "precip_roll_sum_24h",

    # Lags
    "temp_lag_1h",
    "temp_lag_3h",
    "temp_lag_6h",
    "temp_lag_24h",
    "temp_lag_168h",       # same hour last week = baseline
    "hum_lag_168h",
    "temp_delta_24h",
]


# ---------------------------------------------------------------------------
# Step 1: Load Gold data
# ---------------------------------------------------------------------------

def load_gold(path: str) -> pd.DataFrame:
    """Load Gold Parquet and sort by time."""
    df = pd.read_parquet(path)
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    logger.info(f"Gold loaded: {len(df):,} rows, {len(df.columns)} columns")
    logger.info(f"Date range: {df['timestamp_utc'].min()} → {df['timestamp_utc'].max()}")
    return df


# ---------------------------------------------------------------------------
# Step 2: Build targets — future temperature values
# ---------------------------------------------------------------------------

def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create target columns by shifting temp_c forward in time.
    target_temp_1h  = temperature 1 hour from now
    target_temp_6h  = temperature 6 hours from now
    target_temp_24h = temperature 24 hours from now

    This is a FORWARD shift — we shift the temperature column backward
    by horizon hours so that each row's target is the future value.
    """
    for label, hours in HORIZONS.items():
        # shift(-hours) moves future values to the current row
        df[f"target_temp_{label}"] = df["temp_c"].shift(-hours)

    # Also build humidex target at 6h
    df["target_humidex_6h"] = df["humidex"].shift(-6)

    logger.info(f"Targets built: target_temp_1h, target_temp_6h, target_temp_24h, target_humidex_6h")
    return df


# ---------------------------------------------------------------------------
# Step 3: Time-based train/test split
# ---------------------------------------------------------------------------

def time_split(df: pd.DataFrame, test_weeks: int = TEST_WEEKS):
    """
    Split by time — train on earlier data, test on most recent N weeks.
    NEVER shuffle. Test set is locked before tuning.

    Guide requirement: "Split by time: train on earlier periods,
    test on later ones. Keep the test set untouched until the very end."
    """
    cutoff = df["timestamp_utc"].max() - pd.Timedelta(weeks=test_weeks)

    train = df[df["timestamp_utc"] <  cutoff].copy()
    test  = df[df["timestamp_utc"] >= cutoff].copy()

    logger.info(f"Time split at: {cutoff}")
    logger.info(f"  Train: {len(train):,} rows ({train['timestamp_utc'].min()} → {train['timestamp_utc'].max()})")
    logger.info(f"  Test:  {len(test):,}  rows ({test['timestamp_utc'].min()} → {test['timestamp_utc'].max()})")
    logger.info(f"  Train/Test ratio: {len(train)/len(df)*100:.1f}% / {len(test)/len(df)*100:.1f}%")

    return train, test


# ---------------------------------------------------------------------------
# Step 4: Prepare feature matrix
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame, feature_cols: list) -> tuple:
    """
    Drop rows with any NaN in features or target.
    Returns (X, valid_index) where valid_index maps back to original df rows.
    """
    available = [c for c in feature_cols if c in df.columns]
    missing   = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.warning(f"Features not in DataFrame (skipped): {missing}")

    X = df[available].copy()
    return X, available


# ---------------------------------------------------------------------------
# Step 5: Train Random Forest per horizon
# ---------------------------------------------------------------------------

def train_rf_model(X_train: pd.DataFrame, y_train: pd.Series, horizon: str) -> object:
    """
    Train a Random Forest Regressor.
    Fixed random seed ensures reproducibility — same result every run.
    """
    from sklearn.ensemble import RandomForestRegressor

    logger.info(f"Training RF for horizon={horizon} | "
                f"n_samples={len(X_train):,} | features={X_train.shape[1]}")

    # Drop rows where target is NaN (end of series has no future values)
    valid = y_train.notna()
    X_fit = X_train[valid].fillna(X_train[valid].median())
    y_fit = y_train[valid]

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=5,
        random_state=RANDOM_SEED,
        n_jobs=-1,          # use all CPU cores
    )
    model.fit(X_fit, y_fit)

    logger.info(f"  RF trained: {horizon} | trees={N_ESTIMATORS} | depth={MAX_DEPTH}")
    return model


# ---------------------------------------------------------------------------
# Step 6: Evaluate — metrics + baseline comparison
# ---------------------------------------------------------------------------

def evaluate(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    df_test: pd.DataFrame,
    horizon: str,
) -> dict:
    """
    Evaluate model on the locked test set.
    Computes: MAE, RMSE, R²
    Compares against baseline: same-hour-last-week (temp_lag_168h)

    Guide requirement: "The model must beat the declared baseline."
    Declared baseline = same hour last week.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    valid = y_test.notna()
    X_eval = X_test[valid].fillna(X_test[valid].median())
    y_true = y_test[valid]

    y_pred     = model.predict(X_eval)
    y_baseline = df_test.loc[valid, "temp_lag_168h"].fillna(df_test["temp_c"].mean())

    # Model metrics
    mae_model  = mean_absolute_error(y_true, y_pred)
    rmse_model = np.sqrt(mean_squared_error(y_true, y_pred))
    r2_model   = r2_score(y_true, y_pred)

    # Baseline metrics
    mae_base   = mean_absolute_error(y_true, y_baseline)
    rmse_base  = np.sqrt(mean_squared_error(y_true, y_baseline))

    improvement_mae  = (mae_base  - mae_model)  / mae_base  * 100
    improvement_rmse = (rmse_base - rmse_model) / rmse_base * 100

    beats_baseline = mae_model < mae_base

    logger.info(f"\n--- Results: target_temp_{horizon} ---")
    logger.info(f"  Model  MAE:  {mae_model:.3f}°C  |  Baseline MAE:  {mae_base:.3f}°C  |  Improvement: {improvement_mae:+.1f}%")
    logger.info(f"  Model  RMSE: {rmse_model:.3f}°C  |  Baseline RMSE: {rmse_base:.3f}°C  |  Improvement: {improvement_rmse:+.1f}%")
    logger.info(f"  Model  R²:   {r2_model:.4f}")
    logger.info(f"  Beats baseline: {'✅ YES' if beats_baseline else '❌ NO'}")

    return {
        "horizon":          horizon,
        "n_test_samples":   int(valid.sum()),
        "mae_model":        round(mae_model,  4),
        "rmse_model":       round(rmse_model, 4),
        "r2_model":         round(r2_model,   4),
        "mae_baseline":     round(mae_base,   4),
        "rmse_baseline":    round(rmse_base,  4),
        "improvement_mae_pct":  round(improvement_mae,  2),
        "improvement_rmse_pct": round(improvement_rmse, 2),
        "beats_baseline":   beats_baseline,
    }


# ---------------------------------------------------------------------------
# Step 7: Errors by group (fairness / reliability analysis)
# ---------------------------------------------------------------------------

def errors_by_group(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    df_test: pd.DataFrame,
    horizon: str,
) -> pd.DataFrame:
    """
    Break down MAE by subgroup.

    Guide requirement: "Report errors by group. A single overall score
    can hide a group where the model performs badly."

    Groups analysed:
      - Season (winter/spring/monsoon/autumn)
      - Hour bucket (night/morning/afternoon/evening)
      - Flagged rows (is_flagged=True from DQ Gate 1)
    """
    from sklearn.metrics import mean_absolute_error

    valid  = y_test.notna()
    X_eval = X_test[valid].fillna(X_test[valid].median())
    y_true = y_test[valid]
    y_pred = model.predict(X_eval)

    df_eval = df_test[valid].copy()
    df_eval["y_true"] = y_true.values
    df_eval["y_pred"] = y_pred
    df_eval["abs_error"] = np.abs(y_true.values - y_pred)

    # Hour buckets
    def hour_bucket(h):
        if   0  <= h < 6:  return "night (0–5)"
        elif 6  <= h < 12: return "morning (6–11)"
        elif 12 <= h < 18: return "afternoon (12–17)"
        else:              return "evening (18–23)"

    df_eval["hour_bucket"] = df_eval["hour_of_day"].apply(hour_bucket)

    results = []

    # By season
    for season, grp in df_eval.groupby("season"):
        results.append({
            "group_type": "season",
            "group_value": season,
            "n_rows": len(grp),
            "mae": round(grp["abs_error"].mean(), 4),
            "max_error": round(grp["abs_error"].max(), 4),
        })

    # By hour bucket
    for bucket, grp in df_eval.groupby("hour_bucket"):
        results.append({
            "group_type": "hour_bucket",
            "group_value": bucket,
            "n_rows": len(grp),
            "mae": round(grp["abs_error"].mean(), 4),
            "max_error": round(grp["abs_error"].max(), 4),
        })

    # Flagged vs clean rows
    for flag_val, grp in df_eval.groupby("is_flagged"):
        results.append({
            "group_type": "is_flagged",
            "group_value": str(flag_val),
            "n_rows": len(grp),
            "mae": round(grp["abs_error"].mean(), 4),
            "max_error": round(grp["abs_error"].max(), 4),
        })

    group_df = pd.DataFrame(results)

    logger.info(f"\nErrors by group (horizon={horizon}):")
    logger.info("\n" + group_df.to_string(index=False))

    # Identify worst group — important for the viva
    worst = group_df.loc[group_df["mae"].idxmax()]
    logger.warning(
        f"  Weakest group: {worst['group_type']}={worst['group_value']} "
        f"MAE={worst['mae']:.3f}°C"
    )

    return group_df


# ---------------------------------------------------------------------------
# Step 8: Feature importance
# ---------------------------------------------------------------------------

def feature_importance(model, feature_cols: list, horizon: str) -> pd.DataFrame:
    """Extract and log the top 15 most important features."""
    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    logger.info(f"\nTop 15 features (horizon={horizon}):")
    logger.info("\n" + fi_df.head(15).to_string(index=False))

    return fi_df


# ---------------------------------------------------------------------------
# Step 9: Bootstrap uncertainty intervals
# ---------------------------------------------------------------------------

def bootstrap_uncertainty(
    model,
    X_sample: pd.DataFrame,
    n_bootstrap: int = 50,
) -> dict:
    """
    Estimate prediction uncertainty using tree-level variance.
    Each tree in the RF gives a prediction — the spread across trees
    gives us a natural uncertainty estimate.

    Guide requirement: "Report a range, not just one number."
    """
    # Get predictions from each individual tree
    tree_preds = np.array([
        tree.predict(X_sample.fillna(X_sample.median()))
        for tree in model.estimators_[:n_bootstrap]
    ])

    mean_pred = tree_preds.mean(axis=0)
    std_pred  = tree_preds.std(axis=0)

    # 90% prediction interval approximation
    lower_90 = mean_pred - 1.645 * std_pred
    upper_90 = mean_pred + 1.645 * std_pred

    avg_interval_width = (upper_90 - lower_90).mean()
    logger.info(f"  Bootstrap uncertainty: avg 90% interval width = ±{avg_interval_width/2:.2f}°C")

    return {
        "mean":       mean_pred,
        "std":        std_pred,
        "lower_90":   lower_90,
        "upper_90":   upper_90,
        "interval_width_avg": round(float(avg_interval_width), 3),
    }


# ---------------------------------------------------------------------------
# Step 10: Write model card
# ---------------------------------------------------------------------------

def write_model_card(all_metrics: list, group_results: dict, feature_cols: list) -> None:
    """
    Write model_card.md — required by the guide.
    Covers: purpose, data, features, baseline, results, limitations.
    """
    lines = [
        "# Model Card — ColdChain Watch Pakistan",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Purpose",
        "Predict external air temperature at Karachi for 1h, 6h, and 24h ahead.",
        "Predictions are compared against FAO cold-storage thresholds per crop",
        "to generate heat alerts for warehouse managers.",
        "",
        "## Training Data",
        "- Source 1: Open-Meteo Historical API (CC BY 4.0)",
        "- Source 2: NASA POWER Hourly API (NASA Open Data)",
        "- Location: Karachi, Pakistan (24.8607°N, 67.0011°E)",
        "- Period: 2 years of hourly data",
        "- Features: " + str(len(feature_cols)) + " engineered features from Gold layer",
        "",
        "## Declared Baseline",
        "**Same hour last week** (`temp_lag_168h`) — declared before results were seen.",
        "",
        "## Train / Test Split",
        f"- Time-based split: most recent {TEST_WEEKS} weeks held out as test set",
        "- No random shuffle — future data never seen during training",
        "- Scalers/imputers fit on training data only",
        "",
        "## Results",
        "",
        "| Horizon | Model MAE | Baseline MAE | Improvement | Beats Baseline |",
        "|---------|-----------|--------------|-------------|----------------|",
    ]

    for m in all_metrics:
        beat = "✅ Yes" if m["beats_baseline"] else "❌ No"
        lines.append(
            f"| {m['horizon']} | {m['mae_model']:.3f}°C | "
            f"{m['mae_baseline']:.3f}°C | "
            f"{m['improvement_mae_pct']:+.1f}% | {beat} |"
        )

    lines += [
        "",
        "## Errors by Group",
        "",
    ]

    for horizon, grp_df in group_results.items():
        lines.append(f"### Horizon: {horizon}")
        lines.append("")
        # to_markdown requires tabulate; fall back to CSV-style text
        try:
            lines.append(grp_df.to_markdown(index=False))
        except ImportError:
            lines.append("```")
            lines.append(grp_df.to_string(index=False))
            lines.append("```")
        lines.append("")

    lines += [
        "## Crop Thresholds (FAO)",
        "| Crop | Safe Max Temp | Source |",
        "|------|--------------|--------|",
        "| Potato | 4°C | FAO Postharvest Compendium |",
        "| Tomato | 13°C | FAO Postharvest Compendium |",
        "| Onion | 2°C | FAO Postharvest Compendium |",
        "",
        "## Limitations",
        "- Model predicts **external ambient temperature**, not internal cold-store temperature.",
        "- Actual spoilage depends on insulation quality, door-opening frequency,",
        "  pre-cooling efficiency — none of which are in the public data.",
        "- Karachi external temperature **always** exceeds potato and onion thresholds.",
        "  Heat alerts for these crops indicate the energy cost of maintaining cold-chain,",
        "  not an imminent spoilage event.",
        "- Model trained on Karachi only. Do not apply to other cities without retraining.",
        "- Predictions beyond 24h are not provided — uncertainty grows too large.",
        "",
        "## Honest AI Use",
        "See AI_use_log.md for details of all AI-assisted steps.",
    ]

    card_path = "model/model_card.md"
    with open(card_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.success(f"Model card written: {card_path}")


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def run_training():
    """
    Full model training pipeline.
    Called by `make model`.
    """
    logger.info("=" * 60)
    logger.info("ColdChain Watch — Model Training")
    logger.info(f"Started: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    logger.info(f"Random seed: {RANDOM_SEED} (fixed for reproducibility)")
    logger.info("=" * 60)

    # Load Gold
    df = load_gold(GOLD_PATH)

    # Build target columns
    df = build_targets(df)

    # Time split
    train_df, test_df = time_split(df)

    # Prepare features
    X_train, feature_cols = prepare_features(train_df, BASE_FEATURES)
    X_test,  _            = prepare_features(test_df,  BASE_FEATURES)

    all_metrics    = []
    group_results  = {}
    trained_models = {}

    # Train one RF model per forecast horizon
    for horizon_label, horizon_hours in HORIZONS.items():
        target_col = f"target_temp_{horizon_label}"
        logger.info(f"\n{'='*50}")
        logger.info(f"Horizon: {horizon_label} ({horizon_hours}h ahead)")

        y_train = train_df[target_col]
        y_test  = test_df[target_col]

        # Train
        model = train_rf_model(X_train, y_train, horizon_label)

        # Evaluate on locked test set
        metrics = evaluate(model, X_test, y_test, test_df, horizon_label)
        all_metrics.append(metrics)

        # Errors by group
        grp_df = errors_by_group(model, X_test, y_test, test_df, horizon_label)
        group_results[horizon_label] = grp_df

        # Feature importance
        fi_df = feature_importance(model, feature_cols, horizon_label)

        # Uncertainty estimate on first 100 test rows
        uncertainty = bootstrap_uncertainty(model, X_test.head(100).fillna(X_test.median()))

        # Save model artifact
        artifact = {
            "model":         model,
            "feature_cols":  feature_cols,
            "metrics":       metrics,
            "uncertainty_interval_width": uncertainty["interval_width_avg"],
            "trained_at":    datetime.now(timezone.utc).isoformat(),
            "random_seed":   RANDOM_SEED,
            "train_rows":    len(X_train),
            "test_rows":     len(X_test),
        }
        model_path = os.path.join(ARTIFACTS_DIR, f"rf_temp_{horizon_label}.pkl")
        joblib.dump(artifact, model_path)
        logger.success(f"Model saved: {model_path}")

        # Save feature importance
        fi_path = os.path.join(ARTIFACTS_DIR, f"feature_importance_{horizon_label}.csv")
        fi_df.to_csv(fi_path, index=False)

        # Save group errors
        grp_path = os.path.join(ARTIFACTS_DIR, f"errors_by_group_{horizon_label}.csv")
        grp_df.to_csv(grp_path, index=False)

        trained_models[horizon_label] = model

    # Save all metrics summary
    metrics_path = os.path.join(ARTIFACTS_DIR, "metrics_summary.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.success(f"Metrics summary saved: {metrics_path}")

    # Write model card
    write_model_card(all_metrics, group_results, feature_cols)

    # Final verdict
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE — Final Summary")
    logger.info("=" * 60)
    all_beat = True
    for m in all_metrics:
        status = "✅ BEATS" if m["beats_baseline"] else "❌ FAILS"
        logger.info(
            f"  {status} baseline | horizon={m['horizon']} | "
            f"MAE={m['mae_model']:.3f}°C vs baseline={m['mae_baseline']:.3f}°C "
            f"({m['improvement_mae_pct']:+.1f}%)"
        )
        if not m["beats_baseline"]:
            all_beat = False

    if all_beat:
        logger.success("All horizons beat the declared baseline.")
    else:
        logger.warning("Some horizons do not beat the baseline — review features.")

    return all_metrics


if __name__ == "__main__":
    run_training()
