"""Priority helpers and multilingual duplicate detection.

Semantic sentence embeddings are preferred; a TF-IDF fallback keeps the local
prototype dependency-light and makes the dedup feature available immediately.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from src.configuration import load_threshold_settings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


@lru_cache(maxsize=1)
def _get_sentence_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    except Exception:
        return None


def make_embedder(model_name=None):
    model_name = model_name or os.getenv(
        "DEDUP_EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
    )
    return _get_sentence_model(model_name)


def check_duplicate_against_recent(
    new_text: str,
    recent_rows,
    vectorizer=None,
    threshold: float | None = None,
    embedder=None,
):
    """Return ``(duplicate_id, similarity, method)`` for the closest match.

    The same-category/time-window filtering remains in the application layer.
    """
    if threshold is None:
        threshold = load_threshold_settings()["duplicate_similarity_threshold"]
    if not recent_rows:
        return None, 0.0, "none"

    if embedder is not None:
        try:
            all_texts = [new_text] + [row[1] for row in recent_rows]
            vectors = embedder.encode(all_texts, normalize_embeddings=True)
            new_vec = np.asarray(vectors[0])
            sims = [cosine_similarity(new_vec, np.asarray(v)) for v in vectors[1:]]
            best_idx = int(np.argmax(sims))
            best_sim = round(float(sims[best_idx]), 3)
            best_id = recent_rows[best_idx][0]
            return (best_id if best_sim >= threshold else None), best_sim, "sentence-transformer"
        except Exception:
            pass

    if vectorizer is None:
        return None, 0.0, "none"

    new_vec = vectorizer.transform([new_text]).toarray()[0]
    best_id, best_sim = None, 0.0
    for gid, text in recent_rows:
        vec = vectorizer.transform([text]).toarray()[0]
        sim = cosine_similarity(new_vec, vec)
        if sim > best_sim:
            best_sim, best_id = sim, gid
    best_sim = round(best_sim, 3)
    return (best_id if best_sim >= threshold else None), best_sim, "tfidf-fallback"
