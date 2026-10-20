"""
gold.py
-------
DQ Gate 2: Silver → Gold

What this file does, in order:
  1. Load Silver Parquet
  2. DQ Gate 2 checks:
       a. Completeness — fail if >5% of temp_c still missing
       b. Freshness    — flag if latest record is stale
       c. Drift        — warn if recent temp distribution shifted
  3. Feature engineering:
       a. Time features   — hour, day_of_week, month, season, is_weekend
       b. Rolling stats   — temp and humidity rolling mean/max (3h, 6h, 24h)
       c. Lag features    — temp and humidity at -1h, -3h, -24h, -168h (same hour last week)
       d. Heat stress     — feels_like adjusted for humidity (humidex)
       e. Heat alerts     — binary flag per crop (FAO thresholds)
       f. Heat risk score — continuous 0–1 severity per crop
  4. Add stable row_id — required by guide: every Gold row needs an ID
     so the RAG agent can cite specific rows
  5. Write to DuckDB (warehouse) + Parquet (portability)
  6. Write crop thresholds lookup table to Gold

Guide rule: "Every number shown on the dashboard must come from Gold."
Guide rule: "A failed gate stops the run. Gold keeps the last good version."

Run: python -m pipeline.gold
"""

import os
from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd
from loguru import logger

from pipeline.config import (
    CROP_THRESHOLDS,
    GOLD_DIR,
    LOCATIONS,
    SILVER_DIR,
    WAREHOUSE_DB,
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
os.makedirs(GOLD_DIR, exist_ok=True)
os.makedirs("data/logs", exist_ok=True)
logger.add(
    "data/logs/dq_gate2_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)

# Gate 2 thresholds
COMPLETENESS_FAIL_PCT  = 5.0   # fail if >5% of temp_c is missing
FRESHNESS_WARN_HOURS   = 26    # warn if latest record older than 26h
DRIFT_STD_MULTIPLIER   = 3.0   # warn if recent mean > 3 std devs from historical


# ---------------------------------------------------------------------------
# DQ Gate 2a: Completeness check
# ---------------------------------------------------------------------------

def check_completeness(df: pd.DataFrame, city: str) -> bool:
    """
    Fail the gate if more than 5% of primary temperature values are missing.
    If it fails, Gold is NOT updated — the last good version is kept.
    Returns True if passes, False if fails.
    """
    missing_pct = df["temp_c"].isna().mean() * 100

    if missing_pct > COMPLETENESS_FAIL_PCT:
        logger.error(
            f"[GATE2 FAIL] {city} | rule=completeness "
            f"| temp_c missing={missing_pct:.2f}% (threshold={COMPLETENESS_FAIL_PCT}%) "
            f"| action=STOP — Gold NOT updated, keeping last good version"
        )
        return False

    logger.info(
        f"[GATE2 OK] {city} | completeness | temp_c missing={missing_pct:.2f}% — pass"
    )
    return True


# ---------------------------------------------------------------------------
# DQ Gate 2b: Freshness check
# ---------------------------------------------------------------------------

def check_freshness(df: pd.DataFrame, city: str) -> None:
    """
    Warn if the latest record in Silver is older than FRESHNESS_WARN_HOURS.
    Does not stop the pipeline — just flags and logs.
    The dashboard will show a staleness indicator.
    """
    latest    = df["timestamp_utc"].max()
    now_utc   = pd.Timestamp.now(tz="UTC")
    age_hours = (now_utc - latest).total_seconds() / 3600

    if age_hours > FRESHNESS_WARN_HOURS:
        logger.warning(
            f"[GATE2 FLAG] {city} | rule=freshness "
            f"| latest_record={latest} | age={age_hours:.1f}h "
            f"| threshold={FRESHNESS_WARN_HOURS}h — data is stale"
        )
    else:
        logger.info(
            f"[GATE2 OK] {city} | freshness | latest={latest} "
            f"age={age_hours:.1f}h — pass"
        )


# ---------------------------------------------------------------------------
# DQ Gate 2c: Drift detection
# ---------------------------------------------------------------------------

def check_drift(df: pd.DataFrame, city: str) -> None:
    """
    Compare the most recent 7 days mean temperature against the
    full historical mean. If the recent mean is more than
    DRIFT_STD_MULTIPLIER standard deviations away, warn.

    This catches cases where new data looks nothing like training data,
    which would make the model unreliable.
    """
    df_sorted = df.sort_values("timestamp_utc")
    cutoff    = df_sorted["timestamp_utc"].max() - pd.Timedelta(days=7)

    recent     = df_sorted[df_sorted["timestamp_utc"] >= cutoff]["temp_c"].dropna()
    historical = df_sorted[df_sorted["timestamp_utc"] <  cutoff]["temp_c"].dropna()

    if len(recent) < 24 or len(historical) < 168:
        logger.info(f"[GATE2 INFO] {city} | drift check skipped — insufficient data")
        return

    hist_mean = historical.mean()
    hist_std  = historical.std()
    rec_mean  = recent.mean()

    if hist_std == 0:
        return

    z_score = abs(rec_mean - hist_mean) / hist_std

    if z_score > DRIFT_STD_MULTIPLIER:
        logger.warning(
            f"[GATE2 FLAG] {city} | rule=distribution_drift "
            f"| historical_mean={hist_mean:.2f}°C "
            f"| recent_7d_mean={rec_mean:.2f}°C "
            f"| z_score={z_score:.2f} (threshold={DRIFT_STD_MULTIPLIER}) "
            f"| action=flag — model may need retraining"
        )
    else:
        logger.info(
            f"[GATE2 OK] {city} | drift | hist_mean={hist_mean:.2f}°C "
            f"recent_mean={rec_mean:.2f}°C z={z_score:.2f} — pass"
        )


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-based features from the UTC timestamp.
    These are important model inputs — temperature patterns are
    strongly cyclical by hour, day, and season.
    """
    ts = df["timestamp_utc"]

    df["hour_of_day"]  = ts.dt.hour.astype(np.int8)
    df["day_of_week"]  = ts.dt.dayofweek.astype(np.int8)   # 0=Mon, 6=Sun
    df["month"]        = ts.dt.month.astype(np.int8)
    df["day_of_year"]  = ts.dt.dayofyear.astype(np.int16)
    df["is_weekend"]   = ts.dt.dayofweek.isin([5, 6]).astype(np.int8)

    # Season — Pakistan climate seasons
    # Winter: Dec–Feb, Spring: Mar–May, Monsoon: Jun–Sep, Autumn: Oct–Nov
    season_map = {
        12: "winter", 1: "winter",  2: "winter",
        3:  "spring", 4: "spring",  5: "spring",
        6:  "monsoon",7: "monsoon", 8: "monsoon", 9: "monsoon",
        10: "autumn", 11: "autumn",
    }
    df["season"] = df["month"].map(season_map)

    # Cyclical encoding of hour and month — helps tree models see
    # that hour 23 and hour 0 are adjacent, not far apart
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24).round(6)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24).round(6)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).round(6)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).round(6)

    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling statistics over past windows.
    These capture recent heat trends — a warehouse manager cares
    about sustained heat, not just a single spike.

    All windows use min_periods=1 so early rows don't become NaN.
    """
    # Sort by time first — rolling requires chronological order
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    # Temperature rolling features
    df["temp_roll_mean_3h"]  = df["temp_c"].rolling(3,  min_periods=1).mean().round(3)
    df["temp_roll_mean_6h"]  = df["temp_c"].rolling(6,  min_periods=1).mean().round(3)
    df["temp_roll_mean_24h"] = df["temp_c"].rolling(24, min_periods=1).mean().round(3)
    df["temp_roll_max_3h"]   = df["temp_c"].rolling(3,  min_periods=1).max().round(3)
    df["temp_roll_max_6h"]   = df["temp_c"].rolling(6,  min_periods=1).max().round(3)
    df["temp_roll_max_24h"]  = df["temp_c"].rolling(24, min_periods=1).max().round(3)
    df["temp_roll_std_24h"]  = df["temp_c"].rolling(24, min_periods=2).std().round(3)

    # Humidity rolling features
    df["hum_roll_mean_3h"]   = df["humidity_pct"].rolling(3,  min_periods=1).mean().round(3)
    df["hum_roll_mean_6h"]   = df["humidity_pct"].rolling(6,  min_periods=1).mean().round(3)
    df["hum_roll_mean_24h"]  = df["humidity_pct"].rolling(24, min_periods=1).mean().round(3)

    # Precipitation rolling (has a lot of zeros — sum is more useful)
    df["precip_roll_sum_6h"]  = df["precip_mm"].rolling(6,  min_periods=1).sum().round(3)
    df["precip_roll_sum_24h"] = df["precip_mm"].rolling(24, min_periods=1).sum().round(3)

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lag features — values from previous time steps.
    The most important lag for this project is 168h (same hour last week)
    because that is the declared baseline for the heat exposure model.
    """
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    # Short lags (recent trend)
    df["temp_lag_1h"]   = df["temp_c"].shift(1)
    df["temp_lag_3h"]   = df["temp_c"].shift(3)
    df["temp_lag_6h"]   = df["temp_c"].shift(6)
    df["temp_lag_24h"]  = df["temp_c"].shift(24)

    # 168h lag = same hour last week — this IS the declared baseline
    df["temp_lag_168h"] = df["temp_c"].shift(168)
    df["hum_lag_168h"]  = df["humidity_pct"].shift(168)

    # Trend: difference from same hour yesterday
    df["temp_delta_24h"] = (df["temp_c"] - df["temp_lag_24h"]).round(3)

    return df


def add_humidex(df: pd.DataFrame) -> pd.DataFrame:
    """
    Humidex — a heat stress index combining temperature and humidity.
    More relevant for cold-chain management than temperature alone,
    because humidity affects how fast stored produce loses moisture.

    Formula: humidex = temp + 0.5555 * (6.11 * e^(5417.7530*(1/273.16 - 1/(273.16+dewpoint))) - 10)
    Simplified approximation used here for hourly efficiency.
    """
    # Dew point approximation (Magnus formula)
    T  = df["temp_c"]
    RH = df["humidity_pct"]

    # Magnus formula constants
    a, b = 17.625, 243.04
    alpha = np.log(RH / 100.0 + 1e-9) + (a * T) / (b + T)
    dew_point = (b * alpha) / (a - alpha)

    # Humidex
    e = 6.112 * np.exp(5417.7530 * (1.0 / 273.16 - 1.0 / (273.16 + dew_point)))
    df["humidex"] = (T + 0.5555 * (e - 10.0)).round(2)
    df["dew_point_c"] = dew_point.round(2)

    return df


def add_heat_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary heat alert flags per crop based on FAO cold storage thresholds.

    heat_alert_{crop} = 1 means the EXTERNAL temperature exceeds the safe
    storage upper limit for that crop. This is the primary signal for the
    warehouse manager — if outside temp is already at or above storage temp,
    pre-cooling loads increase dramatically.

    Also adds a continuous heat_risk_score (0.0 to 1.0) showing how far
    above the threshold the temperature is, scaled to a 20°C range.
    0.0 = at or below threshold (safe)
    1.0 = 20°C above threshold (extreme)
    """
    for crop, thresholds in CROP_THRESHOLDS.items():
        max_t = thresholds["max_temp_c"]

        # Binary alert
        df[f"heat_alert_{crop}"] = (df["temp_c"] > max_t).astype(np.int8)

        # Continuous risk score
        excess = (df["temp_c"] - max_t).clip(lower=0)
        df[f"heat_risk_{crop}"] = (excess / 20.0).clip(upper=1.0).round(4)

    # Combined alert — 1 if ANY crop threshold is exceeded
    alert_cols = [f"heat_alert_{c}" for c in CROP_THRESHOLDS]
    df["heat_alert_any"] = df[alert_cols].max(axis=1).astype(np.int8)

    return df


def add_row_ids(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """
    Add a stable, unique row_id to every Gold row.
    The guide requires: "every number in an insight comes from a Gold table,
    and row IDs are stored" — the RAG agent cites these IDs.

    Format: KHI_2024101900 for Karachi, 2024-10-19, hour 00
    City codes: KHI=Karachi, LHE=Lahore, MUX=Multan, etc.
    """
    city_codes = {
        "Karachi":    "KHI",
        "Lahore":     "LHE",
        "Multan":     "MUX",
        "Okara":      "OKA",
        "Peshawar":   "PEW",
        "Faisalabad": "FSD",
    }
    code = city_codes.get(city, city[:3].upper())

    # Format: KHI_2024101900
    df["row_id"] = (
        code + "_" +
        df["timestamp_utc"].dt.strftime("%Y%m%d%H")
    )

    return df


# ---------------------------------------------------------------------------
# Gold summary log
# ---------------------------------------------------------------------------

def log_gold_summary(df: pd.DataFrame, city: str) -> None:
    """Print a comprehensive summary of the Gold table for the jury."""
    total       = len(df)
    alerts      = df["heat_alert_any"].sum()
    alert_pct   = 100 * alerts / total

    logger.info("=" * 60)
    logger.info(f"[GOLD SUMMARY] {city}")
    logger.info(f"  Total rows:         {total:,}")
    logger.info(f"  Date range:         {df['timestamp_utc'].min()} → {df['timestamp_utc'].max()}")
    logger.info(f"  Temp range:         {df['temp_c'].min():.1f}°C – {df['temp_c'].max():.1f}°C")
    logger.info(f"  Mean temp:          {df['temp_c'].mean():.2f}°C")
    logger.info(f"  Humidity range:     {df['humidity_pct'].min():.0f}% – {df['humidity_pct'].max():.0f}%")
    logger.info(f"  Heat alerts (any):  {alerts:,} rows ({alert_pct:.1f}%)")
    for crop in CROP_THRESHOLDS:
        crop_alerts = df[f"heat_alert_{crop}"].sum()
        logger.info(f"    heat_alert_{crop}:  {crop_alerts:,} rows ({100*crop_alerts/total:.1f}%)")
    logger.info(f"  Columns:            {len(df.columns)}")
    logger.info(f"  Row ID sample:      {df['row_id'].iloc[0]} → {df['row_id'].iloc[-1]}")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Main Gold pipeline
# ---------------------------------------------------------------------------

def process_city_gold(city: str, con: duckdb.DuckDBPyConnection) -> bool:
    """
    Run full Silver → Gold pipeline for one city.
    Returns True if Gold was updated, False if gate failed.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"DQ Gate 2 — Processing Gold: {city}")
    logger.info(f"{'='*60}")

    silver_path = os.path.join(SILVER_DIR, f"weather_silver_{city.lower()}.parquet")
    if not os.path.exists(silver_path):
        logger.error(f"Silver file not found: {silver_path} — run `make silver` first")
        return False

    df = pd.read_parquet(silver_path)
    logger.info(f"Loaded Silver for {city}: {len(df):,} rows, {len(df.columns)} columns")

    # --- Gate 2a: Completeness ---
    if not check_completeness(df, city):
        return False   # Gate failed — Gold not updated

    # --- Gate 2b: Freshness ---
    check_freshness(df, city)   # warns but doesn't stop

    # --- Gate 2c: Drift ---
    check_drift(df, city)       # warns but doesn't stop

    # --- Feature engineering ---
    logger.info(f"Engineering features for {city}...")
    df = add_time_features(df)
    df = add_rolling_features(df)
    df = add_lag_features(df)
    df = add_humidex(df)
    df = add_heat_alerts(df)
    df = add_row_ids(df, city)

    logger.info(f"Feature engineering complete: {len(df.columns)} columns")

    # --- Write to DuckDB Gold table ---
    table_name = f"weather_gold_{city.lower()}"
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
    logger.success(f"DuckDB Gold table written: {table_name} ({len(df):,} rows)")

    # --- Write Parquet ---
    gold_path = os.path.join(GOLD_DIR, f"weather_gold_{city.lower()}.parquet")
    df.to_parquet(gold_path, index=False)
    logger.success(f"Gold Parquet saved: {gold_path}")

    # --- Summary ---
    log_gold_summary(df, city)

    return True


def run_gold():
    """
    Run DQ Gate 2 for all configured locations.
    Called by `make gold` and `make data`.
    """
    logger.info("=" * 60)
    logger.info("ColdChain Watch — DQ Gate 2: Silver → Gold")
    logger.info(f"Started: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    logger.info("=" * 60)

    con = duckdb.connect(WAREHOUSE_DB)

    # Write crop thresholds as a Gold lookup table
    # This is the FAO reference data — every heat alert links back to this
    thresholds_df = pd.DataFrame([
        {
            "crop":          crop,
            "max_temp_c":    vals["max_temp_c"],
            "min_humidity":  vals["min_humidity"],
            "max_humidity":  vals["max_humidity"],
            "source":        "FAO Postharvest Compendium",
            "source_url":    "https://www.fao.org/3/x5055e/x5055e00.htm",
        }
        for crop, vals in CROP_THRESHOLDS.items()
    ])
    con.execute("DROP TABLE IF EXISTS crop_thresholds")
    con.execute("CREATE TABLE crop_thresholds AS SELECT * FROM thresholds_df")
    logger.success("Crop thresholds lookup table written to Gold (DuckDB)")

    # Process each city
    failed = []
    for location in LOCATIONS:
        city = location["name"]
        success = process_city_gold(city, con)
        if not success:
            failed.append(city)

    con.close()

    if failed:
        logger.error(f"Gold build failed for: {failed}")
        raise RuntimeError(f"DQ Gate 2 failed for: {failed}")

    logger.success("DQ Gate 2 complete — all cities written to Gold.")


if __name__ == "__main__":
    run_gold()
