"""
rag/index.py
------------
Build the TF-IDF vector index for the Insight Agent.

Uses scikit-learn TF-IDF vectorizer — no torch, no DLLs, no compiler needed.
Pure Python, works everywhere.

Vectors + vectorizer stored as joblib files in data/rag_store/.
Metadata stored as JSON alongside.

Two indexes:
  - documents  : FAO/PMD/WFP reference chunks
  - gold_rows  : most recent 168h of Gold table data as sentences

Note: TF-IDF is keyword-based, not semantic. For the hackathon this is
sufficient — our queries are domain-specific enough that keyword overlap
works well. Replace with sentence-transformers once VC++ is installed.

Run: python -m rag.index
"""

import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from pipeline.config import CROP_THRESHOLDS, GOLD_DIR
from rag.documents import get_all_chunks

os.makedirs("data/logs", exist_ok=True)
os.makedirs("data/rag_store", exist_ok=True)

logger.add(
    "data/logs/rag_index_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)

RAG_STORE_DIR = "data/rag_store"
RECENT_ROWS   = 168

DOCS_VECTORIZER_PATH = os.path.join(RAG_STORE_DIR, "docs_vectorizer.pkl")
DOCS_VECTORS_PATH    = os.path.join(RAG_STORE_DIR, "documents_vectors.npy")
DOCS_METADATA_PATH   = os.path.join(RAG_STORE_DIR, "documents_metadata.json")
GOLD_VECTORIZER_PATH = os.path.join(RAG_STORE_DIR, "gold_vectorizer.pkl")
GOLD_VECTORS_PATH    = os.path.join(RAG_STORE_DIR, "gold_vectors.npy")
GOLD_METADATA_PATH   = os.path.join(RAG_STORE_DIR, "gold_metadata.json")


def build_vectorizer(texts: list[str]) -> tuple:
    """Fit TF-IDF and return (vectorizer, L2-normalised dense matrix)."""
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
    )
    matrix = vec.fit_transform(texts).toarray().astype(np.float32)
    matrix = normalize(matrix, norm="l2")
    return vec, matrix


def embed_query(query: str, vectorizer: TfidfVectorizer) -> np.ndarray:
    """Transform a query string using a fitted vectorizer, return L2-normed vector."""
    vec = vectorizer.transform([query]).toarray().astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec[0]


# ---------------------------------------------------------------------------
# Index documents
# ---------------------------------------------------------------------------

def index_documents() -> None:
    logger.info("Indexing reference documents (TF-IDF)...")
    chunks = get_all_chunks()
    texts  = [c["text"] for c in chunks]

    vectorizer, matrix = build_vectorizer(texts)

    metadata = [
        {
            "doc_id":     c["doc_id"],
            "source":     c["source"],
            "source_url": c["source_url"],
            "title":      c["title"],
            "crop":       c["crop"],
            "topic":      c["topic"],
            "text":       c["text"],
        }
        for c in chunks
    ]

    joblib.dump(vectorizer, DOCS_VECTORIZER_PATH)
    np.save(DOCS_VECTORS_PATH, matrix)
    with open(DOCS_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.success(f"Documents indexed: {len(chunks)} chunks, matrix shape {matrix.shape}")


# ---------------------------------------------------------------------------
# Gold row → natural language
# ---------------------------------------------------------------------------

def gold_row_to_text(row: pd.Series) -> str:
    ts_pkt = str(row["timestamp_pkt"])[:16] if pd.notna(row.get("timestamp_pkt")) else "unknown"
    alerts = ", ".join(
        f"{c}={'ALERT' if row.get(f'heat_alert_{c}', 0) == 1 else 'OK'}"
        for c in CROP_THRESHOLDS
    )
    risks = ", ".join(
        f"{c}={row.get(f'heat_risk_{c}', 0):.2f}"
        for c in CROP_THRESHOLDS
    )
    return (
        f"At {row['row_id']} ({row['city']}, {ts_pkt} PKT, {row.get('season','unknown')} season): "
        f"temperature {row['temp_c']:.1f} degrees celsius "
        f"feels like {row['feels_like_c']:.1f} degrees humidex {row.get('humidex', 0):.1f} "
        f"humidity {row['humidity_pct']:.0f} percent "
        f"wind {row.get('wind_kmh', 0):.1f} kmh "
        f"precipitation {row.get('precip_mm', 0):.1f} mm. "
        f"Heat alerts: {alerts}. Heat risk scores: {risks}. "
        f"Rolling mean temperature 6h: {row.get('temp_roll_mean_6h', row['temp_c']):.1f}. "
        f"24h max temperature: {row.get('temp_roll_max_24h', row['temp_c']):.1f}."
    )


# ---------------------------------------------------------------------------
# Index Gold rows
# ---------------------------------------------------------------------------

def index_gold_rows(city: str = "Karachi") -> None:
    logger.info(f"Indexing recent Gold rows for {city} (TF-IDF, last {RECENT_ROWS}h)...")
    gold_path = os.path.join(GOLD_DIR, f"weather_gold_{city.lower()}.parquet")
    if not os.path.exists(gold_path):
        logger.error(f"Gold file not found: {gold_path}")
        return

    df = pd.read_parquet(gold_path)
    df = df.sort_values("timestamp_utc").tail(RECENT_ROWS).reset_index(drop=True)

    texts    = [gold_row_to_text(row) for _, row in df.iterrows()]
    vectorizer, matrix = build_vectorizer(texts)

    metadata = []
    for i, (_, row) in enumerate(df.iterrows()):
        metadata.append({
            "row_id":         row["row_id"],
            "city":           row["city"],
            "timestamp_utc":  str(row["timestamp_utc"]),
            "temp_c":         float(row["temp_c"]),
            "humidity_pct":   float(row["humidity_pct"]),
            "heat_alert_any": int(row["heat_alert_any"]),
            "season":         str(row.get("season", "")),
            "text":           texts[i],
        })

    joblib.dump(vectorizer, GOLD_VECTORIZER_PATH)
    np.save(GOLD_VECTORS_PATH, matrix)
    with open(GOLD_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.success(f"Gold rows indexed: {len(df)} rows, matrix shape {matrix.shape}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_indexing(city: str = "Karachi") -> None:
    logger.info("=" * 60)
    logger.info("ColdChain Watch — RAG Indexing (TF-IDF, no torch required)")
    logger.info(f"Started: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    logger.info("=" * 60)
    index_documents()
    index_gold_rows(city)
    logger.success("RAG index complete.")


if __name__ == "__main__":
    run_indexing()
