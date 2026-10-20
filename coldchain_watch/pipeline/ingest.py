"""
ingest.py
---------
Step 1 of the pipeline: Fetch raw data and land it in Bronze.

What this script does:
1. Fetches 2 years of hourly weather from Open-Meteo Historical API
2. Fetches 2 years of hourly weather from NASA POWER API (validation source)
3. Saves each response as a JSON file in data/bronze/weather/
4. Records every file in manifest.csv with SHA-256 hash, fetch time, licence

The guide rule: Bronze files are NEVER edited after ingestion.
They are the evidence. If you need to fix something, fix it in Silver.

Run this directly:  python -m pipeline.ingest
Or via Makefile:    make data
"""

import json
import os
import time
from datetime import datetime, timezone

import requests
from loguru import logger

from pipeline.config import (
    BRONZE_WEATHER_DIR,
    HISTORY_END,
    HISTORY_START,
    LOCATIONS,
    NASA_POWER_PARAMS,
    NASA_POWER_URL,
    OPEN_METEO_HISTORICAL_URL,
    OPEN_METEO_HOURLY_VARS,
)
from pipeline.manifest import record

# Set up logging — every ingestion action is logged to file AND console
os.makedirs("data/logs", exist_ok=True)
logger.add("data/logs/ingest_{time:YYYY-MM-DD}.log", rotation="1 day", level="INFO")


# ---------------------------------------------------------------------------
# Open-Meteo Historical Weather
# ---------------------------------------------------------------------------

def fetch_openmeteo_historical(location: dict) -> str:
    """
    Fetch 2 years of hourly weather data from Open-Meteo Historical API
    for a single location and save it as a JSON file in Bronze.

    Parameters
    ----------
    location : dict with keys: name, lat, lon, province

    Returns
    -------
    str : path to the saved JSON file
    """
    city   = location["name"]
    lat    = location["lat"]
    lon    = location["lon"]

    logger.info(f"Fetching Open-Meteo historical weather for {city} ...")

    # Build the API request parameters
    params = {
        "latitude":        lat,
        "longitude":       lon,
        "start_date":      HISTORY_START,
        "end_date":        HISTORY_END,
        "hourly":          ",".join(OPEN_METEO_HOURLY_VARS),
        "timezone":        "UTC",               # Always store UTC, convert to PKT later
        "wind_speed_unit": "kmh",
        "temperature_unit":"celsius",
    }

    # Build the query string for the manifest URL record
    query_url = OPEN_METEO_HISTORICAL_URL + "?" + "&".join(
        f"{k}={v}" for k, v in params.items()
    )

    try:
        response = requests.get(OPEN_METEO_HISTORICAL_URL, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Open-Meteo fetch failed for {city}: {e}")
        raise

    # Add metadata to the raw JSON before saving
    # This is extra evidence — who fetched it, when, from where
    data["_meta"] = {
        "source":       "Open-Meteo Historical API",
        "publisher":    "Open-Meteo",
        "licence":      "CC BY 4.0",
        "fetch_time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "city":         city,
        "lat":          lat,
        "lon":          lon,
        "history_start": HISTORY_START,
        "history_end":   HISTORY_END,
        "query_url":    query_url,
    }

    # Save to Bronze — filename includes city and date so reruns don't overwrite
    os.makedirs(BRONZE_WEATHER_DIR, exist_ok=True)
    filename = f"openmeteo_historical_{city.lower()}_{HISTORY_START}_{HISTORY_END}.json"
    filepath = os.path.join(BRONZE_WEATHER_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    logger.success(f"Saved Open-Meteo historical for {city} → {filepath}")

    # Record in manifest.csv
    record(
        source_name=f"Open-Meteo Historical Weather — {city}",
        url=query_url,
        local_filepath=filepath,
        licence="CC BY 4.0",
        notes=f"Hourly weather, {HISTORY_START} to {HISTORY_END}, UTC timezone",
    )

    return filepath


# ---------------------------------------------------------------------------
# NASA POWER API
# ---------------------------------------------------------------------------

def fetch_nasa_power(location: dict) -> str:
    """
    Fetch 2 years of hourly climate data from NASA POWER API.
    Used as a second independent source and for validation against Open-Meteo.

    NASA POWER covers temperature, humidity, and precipitation globally
    using satellite and reanalysis data — it is independent of Open-Meteo.

    Parameters
    ----------
    location : dict with keys: name, lat, lon, province

    Returns
    -------
    str : path to the saved JSON file
    """
    city = location["name"]
    lat  = location["lat"]
    lon  = location["lon"]

    logger.info(f"Fetching NASA POWER data for {city} ...")

    params = {
        "parameters": ",".join(NASA_POWER_PARAMS),
        "community":  "RE",            # Renewable Energy community dataset
        "longitude":  lon,
        "latitude":   lat,
        "start":      HISTORY_START.replace("-", ""),   # NASA format: YYYYMMDD
        "end":        HISTORY_END.replace("-", ""),
        "format":     "JSON",
        "time-standard": "UTC",
    }

    query_url = NASA_POWER_URL + "?" + "&".join(f"{k}={v}" for k, v in params.items())

    try:
        # NASA POWER can be slow — give it 120 seconds
        response = requests.get(NASA_POWER_URL, params=params, timeout=120)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"NASA POWER fetch failed for {city}: {e}")
        raise

    # Add metadata
    data["_meta"] = {
        "source":        "NASA POWER Hourly API",
        "publisher":     "NASA Langley Research Center",
        "licence":       "NASA Open Data — https://power.larc.nasa.gov/docs/services/api/",
        "fetch_time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "city":          city,
        "lat":           lat,
        "lon":           lon,
        "history_start": HISTORY_START,
        "history_end":   HISTORY_END,
        "query_url":     query_url,
    }

    os.makedirs(BRONZE_WEATHER_DIR, exist_ok=True)
    filename = f"nasa_power_{city.lower()}_{HISTORY_START}_{HISTORY_END}.json"
    filepath = os.path.join(BRONZE_WEATHER_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    logger.success(f"Saved NASA POWER data for {city} → {filepath}")

    # Record in manifest.csv
    record(
        source_name=f"NASA POWER Hourly Climate — {city}",
        url=query_url,
        local_filepath=filepath,
        licence="NASA Open Data — https://power.larc.nasa.gov/docs/services/api/",
        notes=f"T2M, RH2M, PRECTOTCORR — {HISTORY_START} to {HISTORY_END}, UTC",
    )

    return filepath


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_ingestion():
    """
    Run full ingestion for all configured locations.
    Called by the Makefile `make data` command.
    """
    logger.info("=" * 60)
    logger.info("ColdChain Watch — Bronze Ingestion Started")
    logger.info(f"Locations: {[loc['name'] for loc in LOCATIONS]}")
    logger.info(f"Date range: {HISTORY_START} to {HISTORY_END}")
    logger.info("=" * 60)

    results = {
        "openmeteo": [],
        "nasa_power": [],
        "failed": [],
    }

    for location in LOCATIONS:
        city = location["name"]

        # --- Open-Meteo ---
        try:
            path = fetch_openmeteo_historical(location)
            results["openmeteo"].append(path)
            time.sleep(1)   # Be polite to the free API
        except Exception as e:
            logger.error(f"Open-Meteo failed for {city}: {e}")
            results["failed"].append(f"openmeteo_{city}")

        # --- NASA POWER ---
        try:
            path = fetch_nasa_power(location)
            results["nasa_power"].append(path)
            time.sleep(2)   # NASA POWER is slower, give it more breathing room
        except Exception as e:
            logger.error(f"NASA POWER failed for {city}: {e}")
            results["failed"].append(f"nasa_power_{city}")

    # Summary
    logger.info("=" * 60)
    logger.info("Ingestion Summary")
    logger.info(f"  Open-Meteo files: {len(results['openmeteo'])}")
    logger.info(f"  NASA POWER files: {len(results['nasa_power'])}")
    logger.info(f"  Failed:           {len(results['failed'])}")
    if results["failed"]:
        logger.warning(f"  Failed items: {results['failed']}")
    logger.info("=" * 60)

    if results["failed"]:
        raise RuntimeError(
            f"Ingestion completed with {len(results['failed'])} failures: "
            f"{results['failed']}"
        )

    logger.success("Bronze ingestion complete. manifest.csv updated.")
    return results


if __name__ == "__main__":
    run_ingestion()
