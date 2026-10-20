"""
manifest.py
-----------
Records every raw file fetched into Bronze.

The hackathon guide requires manifest.csv to contain:
  - source name
  - query / URL used
  - fetch_time (UTC)
  - sha256 hash of the file
  - licence

This module provides two functions:
  - compute_sha256(filepath)  → hash a file
  - record(...)               → append one row to manifest.csv
"""

import csv
import hashlib
import os
from datetime import datetime, timezone

from pipeline.config import MANIFEST_FILE


def compute_sha256(filepath: str) -> str:
    """
    Compute SHA-256 hash of a file.
    The guide requires this to verify file integrity on rebuild.
    """
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read in 64KB chunks to handle large files without loading all into RAM
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def record(
    source_name: str,
    url: str,
    local_filepath: str,
    licence: str,
    notes: str = "",
) -> None:
    """
    Append one entry to manifest.csv.

    Parameters
    ----------
    source_name    : human-readable name, e.g. "Open-Meteo Historical Weather"
    url            : exact URL or API query string used to fetch the data
    local_filepath : path where the raw file was saved in Bronze
    licence        : licence string, e.g. "CC BY 4.0"
    notes          : optional extra context
    """
    # Compute hash of the saved file
    sha256 = compute_sha256(local_filepath)
    fetch_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Create manifest file with header if it doesn't exist yet
    file_exists = os.path.isfile(MANIFEST_FILE)
    os.makedirs(os.path.dirname(MANIFEST_FILE), exist_ok=True)

    with open(MANIFEST_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "source_name", "url", "local_filepath",
            "fetch_time_utc", "sha256", "licence", "notes"
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "source_name":    source_name,
            "url":            url,
            "local_filepath": local_filepath,
            "fetch_time_utc": fetch_time,
            "sha256":         sha256,
            "licence":        licence,
            "notes":          notes,
        })

    print(f"[manifest] Recorded: {source_name} → {local_filepath} (sha256: {sha256[:12]}...)")


def verify_all() -> bool:
    """
    Re-hash every file in manifest.csv and confirm it matches the stored hash.
    Called by make data to prove nothing was manually edited.
    Returns True if all files pass, False if any mismatch found.
    """
    if not os.path.isfile(MANIFEST_FILE):
        print("[manifest] No manifest.csv found — nothing to verify.")
        return False

    all_ok = True
    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filepath = row["local_filepath"]
            expected = row["sha256"]
            if not os.path.isfile(filepath):
                print(f"[manifest] MISSING: {filepath}")
                all_ok = False
                continue
            actual = compute_sha256(filepath)
            if actual != expected:
                print(f"[manifest] HASH MISMATCH: {filepath}")
                print(f"           Expected: {expected}")
                print(f"           Got:      {actual}")
                all_ok = False
            else:
                print(f"[manifest] OK: {filepath}")

    return all_ok
