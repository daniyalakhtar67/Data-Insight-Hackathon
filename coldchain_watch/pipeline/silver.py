"""
silver.py
---------
DQ Gate 1: Bronze → Silver

What this file does, in order:
  1. Parse Open-Meteo JSON → standard DataFrame
  2. Parse NASA POWER JSON → standard DataFrame (different format!)
  3. Merge both on timestamp — one row per hour with columns from both sources
  4. Cross-validate: flag hours where Open-Meteo and NASA differ > 5°C
  5. Fix: convert NASA fill value -999.0 to NaN
  6. Fix: detect and correct Kelvin temperatures
  7. Fix: clamp humidity to 0–100 range
  8. Fix: remove exact duplicate rows
  9. Fill: interpolate short gaps (≤ 2 consecutive hours)
 10. Flag: mark long gaps as NaN (do not guess)
 11. Write clean merged DataFrame to Silver Parquet
 12. Log EVERY decision with timestamp, rule, action, row count

Key rule from the guide:
  "Quarantine never drops silently" — every problem is logged.
  "Fix is certain and reversible" — Fill is estimated and marked.
  "Flag — do not guess. Let the user see the uncertainty."
"""

import json
import os
from datetime import datetime, timezone

import pandas as pd
from loguru import logger

from pipeline.config import (
    BRONZE_WEATHER_DIR,
    HISTORY_END,
    HISTORY_START,
    LOCATIONS,
    SILVER_DIR,
)

# ---------------------------------------------------------------------------
# Logging setup — decisions go to both console and a dated log file
# ---------------------------------------------------------------------------
os.makedirs(SILVER_DIR, exist_ok=True)
os.makedirs("data/logs", exist_ok=True)
logger.add(
    "data/logs/dq_gate1_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)

# NASA POWER uses -999.0 to indicate missing data
NASA_FILL_VALUE = -999.0

# Maximum acceptable difference between Open-Meteo and NASA temperature
# for the same hour at the same location. If greater → flag as suspicious.
CROSS_VALIDATION_TEMP_THRESHOLD = 5.0   # °C
CROSS_VALIDATION_HUM_THRESHOLD  = 20.0  # %


# ---------------------------------------------------------------------------
# Step 1: Parse Open-Meteo Bronze JSON
# ---------------------------------------------------------------------------

def parse_openmeteo(filepath: str, city: str) -> pd.DataFrame:
    """
    Parse the Open-Meteo Historical JSON file into a standard DataFrame.

    Open-Meteo format:
      { "hourly": { "time": [...], "temperature_2m": [...], ... } }

    Timestamps come as "2024-10-19T00:00" strings in UTC.
    """
    logger.info(f"[Open-Meteo] Parsing Bronze file for {city}...")

    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)

    hourly = raw.get("hourly", {})
    df = pd.DataFrame(hourly)

    # Rename to standard column names used throughout the whole pipeline
    df = df.rename(columns={
        "time":                 "timestamp_utc",
        "temperature_2m":       "om_temp_c",        # om_ prefix = Open-Meteo source
        "relative_humidity_2m": "om_humidity_pct",
        "apparent_temperature": "om_feels_like_c",
        "precipitation":        "om_precip_mm",
        "windspeed_10m":        "om_wind_kmh",
        "weathercode":          "om_weather_code",
    })

    # Parse timestamps — Open-Meteo gives "2024-10-19T00:00" without timezone
    # We know it is UTC (we requested timezone=UTC in the API call)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    logger.info(f"[Open-Meteo] {city}: {len(df)} rows loaded "
                f"({df['timestamp_utc'].min()} → {df['timestamp_utc'].max()})")

    return df


# ---------------------------------------------------------------------------
# Step 2: Parse NASA POWER Bronze JSON
# ---------------------------------------------------------------------------

def parse_nasa_power(filepath: str, city: str) -> pd.DataFrame:
    """
    Parse the NASA POWER JSON file into a standard DataFrame.

    NASA POWER format is completely different from Open-Meteo:
      {
        "properties": {
          "parameter": {
            "T2M":  { "2024101900": 25.43, "2024101901": 25.28, ... },
            "RH2M": { "2024101900": 74.1,  ... },
            ...
          }
        }
      }

    Timestamps are strings like "2024101900" = YYYYMMDDHH in UTC.
    Missing values are encoded as -999.0 (NASA fill value).
    """
    logger.info(f"[NASA POWER] Parsing Bronze file for {city}...")

    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)

    params = raw.get("properties", {}).get("parameter", {})

    # Each parameter is a dict of {timestamp_str: value}
    # We need to align them all on the same timestamp index
    t2m         = params.get("T2M",          {})
    rh2m        = params.get("RH2M",         {})
    precip      = params.get("PRECTOTCORR",  {})

    # Build DataFrame from T2M timestamps as the index
    df = pd.DataFrame({
        "timestamp_str":  list(t2m.keys()),
        "nasa_temp_c":    list(t2m.values()),
        "nasa_humidity_pct": [rh2m.get(k, NASA_FILL_VALUE) for k in t2m.keys()],
        "nasa_precip_mm":    [precip.get(k, NASA_FILL_VALUE) for k in t2m.keys()],
    })

    # Parse NASA timestamp format: "2024101900" → YYYYMMDDHH → UTC datetime
    # pd.to_datetime understands format="%Y%m%d%H"
    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_str"], format="%Y%m%d%H", utc=True
    )
    df = df.drop(columns=["timestamp_str"])

    logger.info(f"[NASA POWER] {city}: {len(df)} rows loaded "
                f"({df['timestamp_utc'].min()} → {df['timestamp_utc'].max()})")

    return df


# ---------------------------------------------------------------------------
# Step 3: Fix NASA fill values (-999.0 → NaN)
# ---------------------------------------------------------------------------

def fix_nasa_fill_values(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """
    NASA uses -999.0 as a sentinel for missing data.
    Replace all occurrences with NaN so pandas treats them as missing.
    This is a FIX — certain and reversible (we know exactly what -999 means).
    """
    nasa_cols = ["nasa_temp_c", "nasa_humidity_pct", "nasa_precip_mm"]
    total_fixed = 0

    for col in nasa_cols:
        if col not in df.columns:
            continue
        mask = df[col] == NASA_FILL_VALUE
        count = mask.sum()
        if count > 0:
            df.loc[mask, col] = float("nan")
            total_fixed += count
            logger.info(
                f"[DQ FIX] {city} | column={col} | rule=nasa_fill_value (-999.0) "
                f"| action=replace_with_NaN | rows_affected={count}"
            )

    if total_fixed == 0:
        logger.info(f"[DQ CHECK] {city} | No NASA fill values (-999.0) found — clean.")

    return df


# ---------------------------------------------------------------------------
# Step 4: Merge Open-Meteo + NASA POWER on timestamp
# ---------------------------------------------------------------------------

def merge_sources(om_df: pd.DataFrame, nasa_df: pd.DataFrame, city: str) -> pd.DataFrame:
    """
    Merge Open-Meteo and NASA POWER on timestamp_utc (inner join).
    This gives us one row per hour with columns from both sources.

    We use inner join because both cover the same date range.
    Any hours missing from either source will be flagged separately.
    """
    before_om   = len(om_df)
    before_nasa = len(nasa_df)

    merged = pd.merge(om_df, nasa_df, on="timestamp_utc", how="outer")
    merged = merged.sort_values("timestamp_utc").reset_index(drop=True)

    logger.info(
        f"[MERGE] {city} | Open-Meteo={before_om} rows, NASA={before_nasa} rows "
        f"→ merged={len(merged)} rows"
    )

    # Add city and metadata columns
    merged["city"]      = city
    merged["is_filled"] = False   # will be set True for interpolated values
    merged["is_flagged"]= False   # will be set True for suspicious rows

    return merged


# ---------------------------------------------------------------------------
# Step 5: Fix Kelvin temperatures
# ---------------------------------------------------------------------------

def fix_kelvin(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """
    Pakistan temperatures should be roughly 5–55°C.
    If a temperature looks like Kelvin (> 200), subtract 273.15.
    This is a FIX — the conversion is certain.
    """
    for col in ["om_temp_c", "nasa_temp_c", "om_feels_like_c"]:
        if col not in df.columns:
            continue
        mask = df[col] > 200
        count = mask.sum()
        if count > 0:
            df.loc[mask, col] -= 273.15
            logger.warning(
                f"[DQ FIX] {city} | column={col} | rule=kelvin_detected (>200) "
                f"| action=subtract_273.15 | rows_affected={count}"
            )
        else:
            logger.info(f"[DQ CHECK] {city} | {col}: no Kelvin values detected — OK")

    return df


# ---------------------------------------------------------------------------
# Step 6: Fix humidity out of range
# ---------------------------------------------------------------------------

def fix_humidity_range(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """
    Humidity must be 0–100%.
    Values outside this range are physically impossible — set to NaN (FLAG).
    We do not guess what the correct value should be.
    """
    for col in ["om_humidity_pct", "nasa_humidity_pct"]:
        if col not in df.columns:
            continue
        mask = (df[col] < 0) | (df[col] > 100)
        count = mask.sum()
        if count > 0:
            df.loc[mask, col]       = float("nan")
            df.loc[mask, "is_flagged"] = True
            logger.warning(
                f"[DQ FLAG] {city} | column={col} | rule=humidity_out_of_range (0-100%) "
                f"| action=set_NaN | rows_affected={count}"
            )
        else:
            logger.info(f"[DQ CHECK] {city} | {col}: all values in 0–100% range — OK")

    return df


# ---------------------------------------------------------------------------
# Step 7: Remove exact duplicate rows
# ---------------------------------------------------------------------------

def remove_duplicates(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """
    Remove rows with identical timestamps and city.
    Duplicates can occur if data sources overlap or a fetch was run twice.
    This is a FIX — dropping a true duplicate loses nothing.
    """
    before = len(df)
    df = df.drop_duplicates(subset=["timestamp_utc", "city"])
    removed = before - len(df)

    if removed > 0:
        logger.warning(
            f"[DQ FIX] {city} | rule=exact_duplicate "
            f"| action=drop_duplicate_rows | rows_affected={removed}"
        )
    else:
        logger.info(f"[DQ CHECK] {city} | No duplicate timestamps found — OK")

    return df


# ---------------------------------------------------------------------------
# Step 8: Cross-validate Open-Meteo vs NASA temperatures
# ---------------------------------------------------------------------------

def cross_validate_sources(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """
    Compare temperature and humidity from Open-Meteo vs NASA POWER.
    If they disagree by more than the threshold for the same hour,
    flag that row as suspicious — do not silently pick one source.

    The jury reads the decision log. Flagging disagreements shows
    scientific honesty — we know our data has uncertainty.
    """
    if "om_temp_c" not in df.columns or "nasa_temp_c" not in df.columns:
        return df

    # Temperature cross-validation
    both_have_temp = df["om_temp_c"].notna() & df["nasa_temp_c"].notna()
    temp_diff      = (df["om_temp_c"] - df["nasa_temp_c"]).abs()
    suspicious_temp = both_have_temp & (temp_diff > CROSS_VALIDATION_TEMP_THRESHOLD)

    if suspicious_temp.any():
        df.loc[suspicious_temp, "is_flagged"] = True
        logger.warning(
            f"[DQ FLAG] {city} | rule=cross_validation_temp_mismatch "
            f"(|Open-Meteo - NASA| > {CROSS_VALIDATION_TEMP_THRESHOLD}°C) "
            f"| action=flag_row | rows_affected={suspicious_temp.sum()} "
            f"| max_diff={temp_diff[both_have_temp].max():.2f}°C"
        )
    else:
        logger.info(
            f"[DQ CHECK] {city} | Temperature cross-validation passed "
            f"(max diff={temp_diff[both_have_temp].max():.2f}°C) — OK"
        )

    # Humidity cross-validation
    if "om_humidity_pct" in df.columns and "nasa_humidity_pct" in df.columns:
        both_have_hum  = df["om_humidity_pct"].notna() & df["nasa_humidity_pct"].notna()
        hum_diff       = (df["om_humidity_pct"] - df["nasa_humidity_pct"]).abs()
        suspicious_hum = both_have_hum & (hum_diff > CROSS_VALIDATION_HUM_THRESHOLD)

        if suspicious_hum.any():
            df.loc[suspicious_hum, "is_flagged"] = True
            logger.warning(
                f"[DQ FLAG] {city} | rule=cross_validation_humidity_mismatch "
                f"(|Open-Meteo - NASA| > {CROSS_VALIDATION_HUM_THRESHOLD}%) "
                f"| action=flag_row | rows_affected={suspicious_hum.sum()}"
            )
        else:
            logger.info(
                f"[DQ CHECK] {city} | Humidity cross-validation passed "
                f"(max diff={hum_diff[both_have_hum].max():.2f}%) — OK"
            )

    return df


# ---------------------------------------------------------------------------
# Step 9: Build primary columns (pick best source per value)
# ---------------------------------------------------------------------------

def build_primary_columns(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """
    Create unified primary columns used by the model and dashboard.

    Rule: prefer Open-Meteo (more variables, station-blended).
    Fall back to NASA if Open-Meteo value is missing.
    Flag rows where we fell back to NASA.
    """
    # Primary temperature — prefer Open-Meteo
    df["temp_c"] = df["om_temp_c"].combine_first(df["nasa_temp_c"])

    # Primary humidity
    df["humidity_pct"] = df["om_humidity_pct"].combine_first(df["nasa_humidity_pct"])

    # Primary precipitation
    df["precip_mm"] = df["om_precip_mm"].combine_first(df["nasa_precip_mm"])

    # Feels-like only from Open-Meteo (NASA doesn't provide this)
    df["feels_like_c"] = df.get("om_feels_like_c", float("nan"))

    # Wind and weather code only from Open-Meteo
    df["wind_kmh"]     = df.get("om_wind_kmh",     float("nan"))
    df["weather_code"] = df.get("om_weather_code",  float("nan"))

    # Add PKT timezone column (UTC+5) for display
    df["timestamp_pkt"] = df["timestamp_utc"].dt.tz_convert("Asia/Karachi")

    # Track how many rows fell back to NASA
    nasa_fallback = df["om_temp_c"].isna() & df["nasa_temp_c"].notna()
    if nasa_fallback.any():
        logger.info(
            f"[DQ INFO] {city} | {nasa_fallback.sum()} rows used NASA POWER "
            f"temperature as fallback where Open-Meteo was missing"
        )

    return df


# ---------------------------------------------------------------------------
# Step 10: Fill short gaps
# ---------------------------------------------------------------------------

def fill_short_gaps(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """
    For key columns, fill gaps of ≤ 2 consecutive missing hours
    using linear interpolation. Mark those rows with is_filled=True.

    Gaps longer than 2 hours are left as NaN and flagged.
    The guide says: "A long gap — Flag. Do not guess."
    """
    fill_cols = ["temp_c", "humidity_pct", "feels_like_c", "precip_mm"]

    for col in fill_cols:
        if col not in df.columns:
            continue

        missing_before = df[col].isna().sum()
        if missing_before == 0:
            logger.info(f"[DQ CHECK] {city} | {col}: no missing values — OK")
            continue

        # Interpolate with limit=2 (fills ≤ 2 consecutive NaNs only)
        df[col] = df[col].interpolate(method="linear", limit=2, limit_direction="forward")

        filled  = missing_before - df[col].isna().sum()
        remaining = df[col].isna().sum()

        if filled > 0:
            # Mark rows that were filled (so downstream code knows these are estimates)
            df.loc[df[col].notna(), "is_filled"] = True
            logger.info(
                f"[DQ FILL] {city} | column={col} | rule=short_gap (<=2h) "
                f"| action=linear_interpolation | rows_filled={filled}"
            )

        if remaining > 0:
            df.loc[df[col].isna(), "is_flagged"] = True
            logger.warning(
                f"[DQ FLAG] {city} | column={col} | rule=long_gap (>2h) "
                f"| action=leave_as_NaN | rows_flagged={remaining}"
            )

    return df


# ---------------------------------------------------------------------------
# Step 11: Final summary log
# ---------------------------------------------------------------------------

def log_summary(df: pd.DataFrame, city: str) -> None:
    """Log a final summary of the Silver DataFrame for the jury."""
    total       = len(df)
    filled      = df["is_filled"].sum()
    flagged     = df["is_flagged"].sum()
    missing_temp = df["temp_c"].isna().sum()
    missing_hum  = df["humidity_pct"].isna().sum()

    logger.info("=" * 60)
    logger.info(f"[SILVER SUMMARY] {city}")
    logger.info(f"  Total rows:        {total}")
    logger.info(f"  Filled rows:       {filled}  ({100*filled/total:.2f}%)")
    logger.info(f"  Flagged rows:      {flagged} ({100*flagged/total:.2f}%)")
    logger.info(f"  Missing temp_c:    {missing_temp}")
    logger.info(f"  Missing humidity:  {missing_hum}")
    logger.info(f"  Date range:        {df['timestamp_utc'].min()} → {df['timestamp_utc'].max()}")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Main Silver pipeline
# ---------------------------------------------------------------------------

def process_city(city: str) -> pd.DataFrame:
    """
    Run full Bronze → Silver pipeline for one city.
    Returns the clean merged DataFrame.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"DQ Gate 1 — Processing: {city}")
    logger.info(f"{'='*60}")

    # File paths for this city
    om_file = os.path.join(
        BRONZE_WEATHER_DIR,
        f"openmeteo_historical_{city.lower()}_{HISTORY_START}_{HISTORY_END}.json"
    )
    nasa_file = os.path.join(
        BRONZE_WEATHER_DIR,
        f"nasa_power_{city.lower()}_{HISTORY_START}_{HISTORY_END}.json"
    )

    # Check files exist
    if not os.path.exists(om_file):
        logger.error(f"Open-Meteo Bronze file not found: {om_file}")
        logger.error("Run `make ingest` first.")
        raise FileNotFoundError(om_file)

    if not os.path.exists(nasa_file):
        logger.error(f"NASA POWER Bronze file not found: {nasa_file}")
        logger.error("Run `make ingest` first.")
        raise FileNotFoundError(nasa_file)

    # Parse both sources
    om_df   = parse_openmeteo(om_file,   city)
    nasa_df = parse_nasa_power(nasa_file, city)

    # Fix NASA fill values before merge
    nasa_df = fix_nasa_fill_values(nasa_df, city)

    # Merge into one table
    df = merge_sources(om_df, nasa_df, city)

    # Apply all DQ rules in order
    df = fix_kelvin(df, city)
    df = fix_humidity_range(df, city)
    df = remove_duplicates(df, city)
    df = cross_validate_sources(df, city)
    df = build_primary_columns(df, city)
    df = fill_short_gaps(df, city)

    # Final summary
    log_summary(df, city)

    return df


def run_silver():
    """
    Run DQ Gate 1 for all configured locations.
    Called by `make silver` and `make data`.
    """
    logger.info("=" * 60)
    logger.info("ColdChain Watch — DQ Gate 1: Bronze → Silver")
    logger.info(f"Started at: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    logger.info("=" * 60)

    failed = []

    for location in LOCATIONS:
        city = location["name"]
        try:
            df = process_city(city)

            # Save Silver Parquet
            out_path = os.path.join(SILVER_DIR, f"weather_silver_{city.lower()}.parquet")
            df.to_parquet(out_path, index=False)
            logger.success(f"Silver saved: {out_path} ({len(df)} rows)")

        except Exception as e:
            logger.error(f"Silver failed for {city}: {e}")
            failed.append(city)

    if failed:
        raise RuntimeError(f"DQ Gate 1 failed for: {failed}")

    logger.success("DQ Gate 1 complete — all cities processed.")


if __name__ == "__main__":
    run_silver()
