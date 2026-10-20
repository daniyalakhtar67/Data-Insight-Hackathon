"""
rag/retrieve.py
---------------
TF-IDF cosine similarity retrieval.
Loads vocab from JSON (no pickle) — version-independent.
"""

import json
import os
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from rag.index import (
    DOCS_META_PATH, DOCS_VOCAB_PATH, DOCS_VECTORS_PATH,
    GOLD_META_PATH, GOLD_VOCAB_PATH, GOLD_VECTORS_PATH,
    embed_query, load_vocab,
)

TOP_K_DOCS    = 4
TOP_K_GOLD    = 3
MAX_GOLD_ROWS = 3


@dataclass
class EvidenceBundle:
    query:        str
    doc_chunks:   list[dict] = field(default_factory=list)
    gold_rows:    list[dict] = field(default_factory=list)
    has_evidence: bool       = False

    def to_context_string(self) -> str:
        parts = []
        if self.doc_chunks:
            parts.append("=== REFERENCE DOCUMENTS ===")
            for c in self.doc_chunks:
                parts.append(
                    f"[{c['doc_id']}] {c['title']} "
                    f"(Source: {c['source']}, {c['source_url']})\n{c['text']}"
                )
        if self.gold_rows:
            parts.append("\n=== LIVE DATA FROM GOLD TABLES ===")
            for r in self.gold_rows:
                parts.append(f"[Row ID: {r['row_id']}] {r['text']}")
        return "\n\n".join(parts)

    def citation_list(self) -> list[str]:
        citations = []
        for c in self.doc_chunks:
            citations.append(f"{c['doc_id']}: {c['source']} -- {c['source_url']}")
        for r in self.gold_rows:
            citations.append(f"Gold table row: {r['row_id']}")
        return citations


class InsightRetriever:

    def __init__(self):
        self._doc_vec   = None
        self._doc_mat   = None
        self._doc_meta  = None
        self._gold_vec  = None
        self._gold_mat  = None
        self._gold_meta = None

    def _load_docs(self):
        if self._doc_vec is not None:
            return
        try:
            if os.path.exists(DOCS_VOCAB_PATH):
                self._doc_vec  = load_vocab(DOCS_VOCAB_PATH)
                self._doc_mat  = np.load(DOCS_VECTORS_PATH)
                with open(DOCS_META_PATH, encoding="utf-8") as f:
                    self._doc_meta = json.load(f)
                logger.info(f"Documents index loaded: {len(self._doc_meta)} chunks")
            else:
                logger.warning(f"Documents vocab not found at: {DOCS_VOCAB_PATH}")
        except Exception as e:
            logger.error(f"Failed to load documents index: {e}")
            self._doc_vec = None

    def _load_gold(self):
        if self._gold_vec is not None:
            return
        try:
            if os.path.exists(GOLD_VOCAB_PATH):
                self._gold_vec  = load_vocab(GOLD_VOCAB_PATH)
                self._gold_mat  = np.load(GOLD_VECTORS_PATH)
                with open(GOLD_META_PATH, encoding="utf-8") as f:
                    self._gold_meta = json.load(f)
                logger.info(f"Gold index loaded: {len(self._gold_meta)} rows")
            else:
                logger.warning(f"Gold vocab not found at: {GOLD_VOCAB_PATH}")
        except Exception as e:
            logger.error(f"Failed to load Gold index: {e}")
            self._gold_vec = None

    def _top_k(self, q_vec, matrix, metadata, k):
        scores  = matrix @ q_vec
        top_idx = np.argsort(scores)[::-1][:k]
        results = []
        for i in top_idx:
            item = dict(metadata[i])
            item["score"] = float(scores[i])
            results.append(item)
        return results

    def retrieve(self, query: str) -> EvidenceBundle:
        bundle = EvidenceBundle(query=query)

        self._load_docs()
        if self._doc_vec is not None:
            try:
                q_vec = embed_query(query, self._doc_vec)
                bundle.doc_chunks = self._top_k(
                    q_vec, self._doc_mat, self._doc_meta, TOP_K_DOCS
                )
                logger.info(f"Retrieved {len(bundle.doc_chunks)} doc chunks")
            except Exception as e:
                logger.error(f"Doc retrieval error: {e}")

        self._load_gold()
        if self._gold_vec is not None:
            try:
                q_vec      = embed_query(query, self._gold_vec)
                gold_items = self._top_k(
                    q_vec, self._gold_mat, self._gold_meta, TOP_K_GOLD
                )
                seen_ids = {r["row_id"] for r in gold_items}
                latest   = self._gold_meta[-1]
                if latest["row_id"] not in seen_ids:
                    gold_items.insert(0, dict(latest))
                bundle.gold_rows = gold_items[:MAX_GOLD_ROWS]
                logger.info(f"Retrieved {len(bundle.gold_rows)} Gold rows")
            except Exception as e:
                logger.error(f"Gold retrieval error: {e}")

        bundle.has_evidence = bool(bundle.doc_chunks or bundle.gold_rows)
        return bundle
