"""
config.py
---------
Central configuration for ColdChain Watch Pakistan.
All locations, date ranges, API endpoints, and crop thresholds live here.
Edit this file to add more cities or crops — nothing else needs to change.
"""

from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Locations to monitor
# Start with Karachi only. Add more cities by extending this list.
# ---------------------------------------------------------------------------
LOCATIONS = [
    {
        "name": "Karachi",
        "lat": 24.8607,
        "lon": 67.0011,
        "province": "Sindh",
    },
    # Uncomment to add more cities later:
    # {"name": "Lahore",    "lat": 31.5497, "lon": 74.3436, "province": "Punjab"},
    # {"name": "Multan",    "lat": 30.1575, "lon": 71.5249, "province": "Punjab"},
    # {"name": "Okara",     "lat": 30.8101, "lon": 73.4453, "province": "Punjab"},
    # {"name": "Peshawar",  "lat": 34.0150, "lon": 71.5249, "province": "KPK"},
    # {"name": "Faisalabad","lat": 31.4187, "lon": 73.0791, "province": "Punjab"},
]

# ---------------------------------------------------------------------------
# Historical data window — 2 years back from today
# ---------------------------------------------------------------------------
HISTORY_START = (datetime.utcnow() - timedelta(days=730)).strftime("%Y-%m-%d")
HISTORY_END   = datetime.utcnow().strftime("%Y-%m-%d")

# ---------------------------------------------------------------------------
# Open-Meteo API endpoints (free, no key required)
# Licence: CC BY 4.0 — https://open-meteo.com/en/terms
# ---------------------------------------------------------------------------
OPEN_METEO_HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL   = "https://api.open-meteo.com/v1/forecast"

# Weather variables to fetch (hourly)
OPEN_METEO_HOURLY_VARS = [
    "temperature_2m",         # °C  — air temperature at 2 m height
    "relative_humidity_2m",   # %   — relative humidity
    "apparent_temperature",   # °C  — feels-like temperature
    "precipitation",          # mm  — rainfall per hour
    "windspeed_10m",          # km/h
    "weathercode",            # WMO code — clear/cloudy/rain etc.
]

# ---------------------------------------------------------------------------
# NASA POWER API endpoint (free, no key required)
# Publisher: NASA Langley Research Center
# Licence: https://power.larc.nasa.gov/docs/services/api/
# Used for: daily max/min temperature, solar radiation (backup + validation)
# ---------------------------------------------------------------------------
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"

NASA_POWER_PARAMS = [
    "T2M",       # Temperature at 2 metres (°C)
    "RH2M",      # Relative Humidity at 2 metres (%)
    "PRECTOTCORR",  # Precipitation (mm/hour)
]

# ---------------------------------------------------------------------------
# Crop-specific cold-storage thresholds
# Source: FAO Postharvest Compendium (https://www.fao.org/3/x5055e/x5055e00.htm)
# These are the danger thresholds — exceeding them triggers a heat alert
# ---------------------------------------------------------------------------
CROP_THRESHOLDS = {
    "potato": {
        "max_temp_c":  4.0,    # Store below 4°C
        "min_humidity": 90.0,  # 90–95% RH required
        "max_humidity": 95.0,
    },
    "tomato": {
        "max_temp_c":  13.0,   # Store at 10–13°C
        "min_humidity": 85.0,
        "max_humidity": 90.0,
    },
    "onion": {
        "max_temp_c":  2.0,    # Store at 0–2°C
        "min_humidity": 65.0,
        "max_humidity": 70.0,
    },
}

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
BRONZE_WEATHER_DIR = "data/bronze/weather"
BRONZE_PRICES_DIR  = "data/bronze/prices"
SILVER_DIR         = "data/silver"
GOLD_DIR           = "data/gold"
LOGS_DIR           = "data/logs"
MANIFEST_FILE      = "data/manifest.csv"
WAREHOUSE_DB       = "data/coldchain.duckdb"
