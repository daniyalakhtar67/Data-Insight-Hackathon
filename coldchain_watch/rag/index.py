"""
rag/index.py
------------
Build the TF-IDF vector index for the Insight Agent.

Vectorizer saved as JSON vocab (not pickle) — fully version-independent.
Works on any sklearn version, any Python version, including Streamlit Cloud.

Two indexes:
  - documents  : FAO/PMD/WFP reference chunks
  - gold_rows  : most recent 168h of Gold table data as sentences
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from pipeline.config import CROP_THRESHOLDS, GOLD_DIR
from rag.documents import get_all_chunks

# ---------------------------------------------------------------------------
# Path anchor — relative to coldchain_watch/, not process cwd
# ---------------------------------------------------------------------------
_MODULE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.makedirs(os.path.join(_MODULE_DIR, "data", "logs"),      exist_ok=True)
os.makedirs(os.path.join(_MODULE_DIR, "data", "rag_store"), exist_ok=True)

logger.add(
    os.path.join(_MODULE_DIR, "data", "logs", "rag_index_{time:YYYY-MM-DD}.log"),
    rotation="1 day", level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)

RAG_STORE_DIR = os.path.join(_MODULE_DIR, "data", "rag_store")
RECENT_ROWS   = 168

# Paths — all JSON or npy, NO pickle files
DOCS_VOCAB_PATH   = os.path.join(RAG_STORE_DIR, "docs_vocab.json")
DOCS_VECTORS_PATH = os.path.join(RAG_STORE_DIR, "documents_vectors.npy")
DOCS_META_PATH    = os.path.join(RAG_STORE_DIR, "documents_metadata.json")
GOLD_VOCAB_PATH   = os.path.join(RAG_STORE_DIR, "gold_vocab.json")
GOLD_VECTORS_PATH = os.path.join(RAG_STORE_DIR, "gold_vectors.npy")
GOLD_META_PATH    = os.path.join(RAG_STORE_DIR, "gold_metadata.json")

# Keep old pkl paths as aliases so retrieve.py import doesn't break
DOCS_VECTORIZER_PATH = DOCS_VOCAB_PATH
DOCS_METADATA_PATH   = DOCS_META_PATH
GOLD_VECTORIZER_PATH = GOLD_VOCAB_PATH
GOLD_METADATA_PATH   = GOLD_META_PATH


# ---------------------------------------------------------------------------
# Vectorizer save/load — JSON vocab, no pickle
# ---------------------------------------------------------------------------

def save_vocab(vectorizer: TfidfVectorizer, path: str) -> None:
    """Save TF-IDF vocabulary and IDF weights as plain JSON."""
    vocab_data = {
        "vocabulary":   {k: int(v) for k, v in vectorizer.vocabulary_.items()},
        "idf":          vectorizer.idf_.tolist(),
        "ngram_range":  list(vectorizer.ngram_range),
        "max_features": vectorizer.max_features,
        "sublinear_tf": vectorizer.sublinear_tf,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f)


def load_vocab(path: str) -> TfidfVectorizer:
    """Reconstruct a TF-IDF vectorizer from saved JSON vocab."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    vec = TfidfVectorizer(
        ngram_range=tuple(data["ngram_range"]),
        max_features=data["max_features"],
        sublinear_tf=data["sublinear_tf"],
    )
    vec.vocabulary_ = {k: int(v) for k, v in data["vocabulary"].items()}
    vec.idf_         = np.array(data["idf"], dtype=np.float64)

    # Required internal attributes sklearn needs for transform()
    import scipy.sparse as sp
    n_features = len(vec.vocabulary_)
    vec._tfidf  = vec  # dummy — we use idf_ directly
    # Monkey-patch _tfidf to provide idf_
    from sklearn.feature_extraction.text import TfidfTransformer
    transformer = TfidfTransformer(sublinear_tf=data["sublinear_tf"])
    transformer.idf_ = vec.idf_
    transformer._idf_diag = sp.diags(
        vec.idf_,
        offsets=0,
        shape=(n_features, n_features),
        format="csr",
        dtype=np.float64,
    )
    vec._tfidf = transformer
    return vec


def embed_query(query: str, vectorizer: TfidfVectorizer) -> np.ndarray:
    """Transform a query string → L2-normed vector."""
    v    = vectorizer.transform([query]).toarray().astype(np.float32)
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return v[0]


def build_and_save(texts: list[str], vocab_path: str, vectors_path: str) -> TfidfVectorizer:
    """Fit TF-IDF on texts, save vocab as JSON and vectors as npy."""
    vec    = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
    matrix = vec.fit_transform(texts).toarray().astype(np.float32)
    matrix = normalize(matrix, norm="l2")

    save_vocab(vec, vocab_path)
    np.save(vectors_path, matrix)
    return vec


# ---------------------------------------------------------------------------
# Index documents
# ---------------------------------------------------------------------------

def index_documents() -> None:
    logger.info("Indexing reference documents (TF-IDF, JSON vocab)...")
    chunks = get_all_chunks()
    texts  = [c["text"] for c in chunks]

    build_and_save(texts, DOCS_VOCAB_PATH, DOCS_VECTORS_PATH)

    metadata = [
        {
            "doc_id": c["doc_id"], "source": c["source"],
            "source_url": c["source_url"], "title": c["title"],
            "crop": c["crop"], "topic": c["topic"], "text": c["text"],
        }
        for c in chunks
    ]
    with open(DOCS_META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.success(f"Documents indexed: {len(chunks)} chunks")


# ---------------------------------------------------------------------------
# Gold row → natural language sentence
# ---------------------------------------------------------------------------

def gold_row_to_text(row: pd.Series) -> str:
    ts_pkt = str(row.get("timestamp_pkt", ""))[:16] or "unknown"
    alerts = ", ".join(
        f"{c}={'ALERT' if row.get(f'heat_alert_{c}', 0) == 1 else 'OK'}"
        for c in CROP_THRESHOLDS
    )
    risks = ", ".join(
        f"{c}={row.get(f'heat_risk_{c}', 0.0):.2f}"
        for c in CROP_THRESHOLDS
    )
    return (
        f"At {row['row_id']} ({row['city']}, {ts_pkt} PKT, "
        f"{row.get('season', 'unknown')} season): "
        f"temperature {row['temp_c']:.1f} degrees celsius "
        f"feels like {row.get('feels_like_c', row['temp_c']):.1f} degrees "
        f"humidex {row.get('humidex', 0):.1f} "
        f"humidity {row['humidity_pct']:.0f} percent "
        f"wind {row.get('wind_kmh', 0):.1f} kmh "
        f"precipitation {row.get('precip_mm', 0):.1f} mm. "
        f"Heat alerts: {alerts}. Heat risk scores: {risks}. "
        f"6h mean temp: {row.get('temp_roll_mean_6h', row['temp_c']):.1f}. "
        f"24h max temp: {row.get('temp_roll_max_24h', row['temp_c']):.1f}."
    )


# ---------------------------------------------------------------------------
# Index Gold rows
# ---------------------------------------------------------------------------

def index_gold_rows(city: str = "Karachi") -> None:
    logger.info(f"Indexing Gold rows for {city} (last {RECENT_ROWS}h)...")
    gold_path = os.path.join(GOLD_DIR, f"weather_gold_{city.lower()}.parquet")
    if not os.path.exists(gold_path):
        logger.error(f"Gold file not found: {gold_path}")
        return

    df    = pd.read_parquet(gold_path)
    df    = df.sort_values("timestamp_utc").tail(RECENT_ROWS).reset_index(drop=True)
    texts = [gold_row_to_text(row) for _, row in df.iterrows()]

    build_and_save(texts, GOLD_VOCAB_PATH, GOLD_VECTORS_PATH)

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
    with open(GOLD_META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.success(f"Gold rows indexed: {len(df)} rows")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_indexing(city: str = "Karachi") -> None:
    logger.info("=" * 60)
    logger.info("ColdChain Watch — RAG Indexing (JSON vocab, no pickle)")
    logger.info(f"Started: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    logger.info("=" * 60)
    index_documents()
    index_gold_rows(city)
    logger.success("RAG index complete.")


if __name__ == "__main__":
    run_indexing()
