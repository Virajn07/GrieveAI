"""Similarity search and deterministic recurring-issue grouping."""

from __future__ import annotations

import hashlib
import os
from collections import Counter
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


def _row_values(row):
    if isinstance(row, dict):
        return row.get("id"), row.get("text", ""), row.get("category"), row.get("subcategory")
    if hasattr(row, "id") and hasattr(row, "text"):
        return row.id, row.text, getattr(row, "category", None), getattr(row, "subcategory", None)
    values = tuple(row)
    return (values + (None, None, None, None))[:4]


def _relationship(score: float, duplicate_threshold: float, related_threshold: float) -> str | None:
    if score >= duplicate_threshold:
        return "near_duplicate"
    if score >= related_threshold:
        return "related"
    return None


def find_similar_grievances(
    new_text: str,
    existing_rows,
    *,
    vectorizer=None,
    embedder=None,
    duplicate_threshold: float | None = None,
    related_threshold: float | None = None,
    top_k: int = 5,
):
    """Return the closest exact, near-duplicate or related grievance matches.

    Sentence embeddings are used when supplied. The fitted TF-IDF vectorizer is
    a local/offline fallback; the method is recorded on each result.
    """
    settings = load_threshold_settings()
    duplicate_threshold = (
        settings["duplicate_similarity_threshold"]
        if duplicate_threshold is None else float(duplicate_threshold)
    )
    related_threshold = (
        settings["related_similarity_threshold"]
        if related_threshold is None else float(related_threshold)
    )
    if not 0 <= related_threshold <= duplicate_threshold <= 1:
        raise ValueError("similarity thresholds must satisfy 0 <= related <= duplicate <= 1")
    rows = [_row_values(row) for row in existing_rows]
    rows = [row for row in rows if row[0] is not None and isinstance(row[1], str) and row[1].strip()]
    if not rows:
        return []

    normalized = " ".join(new_text.casefold().split())
    exact = [row for row in rows if " ".join(row[1].casefold().split()) == normalized]
    score_by_index = {}
    method = "sentence-transformer"
    if embedder is not None:
        try:
            vectors = np.asarray(
                embedder.encode([new_text] + [row[1] for row in rows], normalize_embeddings=True)
            )
            score_by_index = {
                index: cosine_similarity(vectors[0], vectors[index + 1])
                for index in range(len(rows))
            }
        except Exception:
            score_by_index = {}
    if not score_by_index and vectorizer is not None:
        try:
            method = "tfidf-fallback"
            vectors = vectorizer.transform([new_text] + [row[1] for row in rows])
            norms = np.sqrt(np.asarray(vectors.multiply(vectors).sum(axis=1)).ravel())
            dots = (vectors[1:] @ vectors[0].T).toarray().ravel()
            score_by_index = {
                index: float(dots[index] / (norms[0] * norms[index + 1] + 1e-9))
                for index in range(len(rows))
            }
        except Exception:
            score_by_index = {}

    exact_indexes = {index for index, row in enumerate(rows) if row in exact}
    for index in exact_indexes:
        score_by_index[index] = 1.0

    results = []
    for index, score in score_by_index.items():
        row_id, old_text, category, subcategory = rows[index]
        kind = "exact_duplicate" if index in exact_indexes else _relationship(
            float(score), duplicate_threshold, related_threshold
        )
        if kind is None:
            continue
        results.append({
            "grievance_id": row_id,
            "similarity": round(float(score), 4),
            "relationship": kind,
            "method": "normalized-text" if kind == "exact_duplicate" else method,
            "category": category,
            "subcategory": subcategory,
            "text": old_text,
        })
    results.sort(key=lambda item: (-item["similarity"], str(item["grievance_id"])))
    return results[:max(0, int(top_k))]


def check_duplicate_against_recent(
    new_text: str,
    recent_rows,
    vectorizer=None,
    threshold: float | None = None,
    embedder=None,
):
    """Legacy app adapter returning ``(duplicate_id, similarity, method)``."""
    matches = find_similar_grievances(
        new_text,
        recent_rows,
        vectorizer=vectorizer,
        embedder=embedder,
        duplicate_threshold=threshold,
        top_k=1,
    )
    if not matches:
        return None, 0.0, "none"
    match = matches[0]
    duplicate_id = match["grievance_id"] if match["relationship"] in {"exact_duplicate", "near_duplicate"} else None
    return duplicate_id, match["similarity"], match["method"]


def cluster_recurring_grievances(
    rows,
    *,
    vectorizer=None,
    embedder=None,
    similarity_threshold: float | None = None,
    min_cluster_size: int = 2,
):
    """Build deterministic connected components of similar grievances.

    This is a first-pass recurring-issue detector, not an LLM labeler. Connected
    components make chained paraphrases group even when their endpoints are
    less similar. The function is intended for small admin windows/datasets.
    """
    settings = load_threshold_settings()
    threshold = settings["recurring_similarity_threshold"] if similarity_threshold is None else float(similarity_threshold)
    if not 0 <= threshold <= 1:
        raise ValueError("recurring similarity threshold must be between 0 and 1")
    parsed = [_row_values(row) for row in rows]
    parsed = [row for row in parsed if row[0] is not None and isinstance(row[1], str) and row[1].strip()]
    if len(parsed) < max(2, int(min_cluster_size)):
        return []

    texts = [row[1] for row in parsed]
    vectors = None
    embedding_vectors = False
    if embedder is not None:
        try:
            vectors = np.asarray(embedder.encode(texts, normalize_embeddings=True))
            embedding_vectors = True
        except Exception:
            vectors = None
    if vectors is None and vectorizer is not None:
        try:
            sparse_vectors = vectorizer.transform(texts)
            norms = np.sqrt(np.asarray(sparse_vectors.multiply(sparse_vectors).sum(axis=1)).ravel())
            similarities = (sparse_vectors @ sparse_vectors.T).toarray()
            vectors = np.divide(
                similarities,
                np.outer(norms, norms) + 1e-9,
                out=np.zeros_like(similarities),
                where=np.outer(norms, norms) > 0,
            )
        except Exception:
            vectors = None
    if vectors is None:
        return []

    parents = list(range(len(parsed)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for left in range(len(parsed)):
        for right in range(left + 1, len(parsed)):
            if " ".join(texts[left].casefold().split()) == " ".join(texts[right].casefold().split()):
                similarity = 1.0
            elif embedding_vectors:
                similarity = cosine_similarity(vectors[left], vectors[right])
            else:
                similarity = float(vectors[left, right])
            if similarity >= threshold:
                root_left, root_right = find(left), find(right)
                if root_left != root_right:
                    parents[root_right] = root_left

    groups = {}
    for index in range(len(parsed)):
        groups.setdefault(find(index), []).append(parsed[index])
    clusters = []
    for members in groups.values():
        if len(members) < min_cluster_size:
            continue
        ids = sorted((str(row[0]) for row in members))
        themes = Counter((row[2], row[3]) for row in members if row[2] or row[3])
        theme = themes.most_common(1)[0][0] if themes else (None, None)
        cluster_key = hashlib.sha1("|".join(ids).encode("utf-8")).hexdigest()[:12]
        clusters.append({
            "cluster_id": f"recurring-{cluster_key}",
            "issue_theme": " / ".join(str(value) for value in theme if value) or "Similar grievance theme",
            "category": theme[0],
            "subcategory": theme[1],
            "grievance_count": len(members),
            "grievance_ids": ids,
            "sample_texts": [row[1] for row in members[:3]],
        })
    return sorted(clusters, key=lambda item: (-item["grievance_count"], item["cluster_id"]))
