"""
Priority estimation (already partly handled by the model's priority_head -
see src/model.py) + near-duplicate detection (Week 3).

Duplicate detection plan:
    - Embed incoming grievance text using the pooled representation exposed
      by GrieveAIClassifier.forward()["pooled"], or a lightweight
      sentence-transformers model if a separate embedder is preferred.
    - Compare against embeddings of grievances submitted in the last N days
      via cosine similarity; flag anything above a threshold (start at 0.85,
      tune against the synthetic near-duplicate examples in
      data/processed/grievances_synthetic.csv where duplicate_of is set).
"""

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def find_duplicates(new_embedding: np.ndarray, recent_embeddings: dict, threshold: float = 0.85):
    """recent_embeddings: {grievance_id: embedding_vector}
    Returns list of (grievance_id, similarity) above threshold, sorted descending."""
    matches = []
    for gid, emb in recent_embeddings.items():
        sim = cosine_similarity(new_embedding, emb)
        if sim >= threshold:
            matches.append((gid, sim))
    return sorted(matches, key=lambda x: -x[1])


def check_duplicate_against_recent(new_text: str, recent_rows, vectorizer, threshold: float = 0.85):
    """Working prototype version - reuses the baseline classifier's own
    TF-IDF vectorizer so no separate embedding model is needed yet.

    recent_rows: list of (grievance_id, text) tuples, e.g. same-category
    grievances submitted in the last 30 days (see app/routes.py).
    Returns (duplicate_id, similarity) - duplicate_id is None if nothing
    clears the threshold, but similarity is always returned so the
    caller/dashboard can show "closest match" even below threshold.

    TODO: once src/model.py is trained, swap `vectorizer.transform` for
    the model's pooled embedding (out["pooled"] in model.py) - same
    downstream cosine_similarity call, better semantic matching."""
    if not recent_rows:
        return None, 0.0

    new_vec = vectorizer.transform([new_text]).toarray()[0]
    best_id, best_sim = None, 0.0
    for gid, text in recent_rows:
        vec = vectorizer.transform([text]).toarray()[0]
        sim = cosine_similarity(new_vec, vec)
        if sim > best_sim:
            best_sim, best_id = sim, gid

    if best_sim >= threshold:
        return best_id, round(best_sim, 3)
    return None, round(best_sim, 3)
