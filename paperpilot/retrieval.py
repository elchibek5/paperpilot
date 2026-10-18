"""
TF-IDF retrieval over chunk texts. Top-k chunks for a query.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# TF-IDF search with scores: (chunk, cosine similarity), best first
def rank_chunks_tfidf(
    chunks: list[dict],
    query: str,
    top_k: int = 5,
) -> list[tuple[dict, float]]:
    """
    Rank chunks by TF-IDF cosine similarity to query; return the top_k as (chunk, score).
    chunks: list of {chunk_id, text, ...}. An empty query returns the first top_k with score 0.
    """
    if not chunks:
        return []
    if not (query and query.strip()):
        return [(c, 0.0) for c in chunks[:top_k]]
    query = query.strip()
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer()
    try:
        docs_tfidf = vectorizer.fit_transform(texts)
    except ValueError:  # no indexable words at all ("empty vocabulary")
        return []
    query_tfidf = vectorizer.transform([query])
    sims = cosine_similarity(query_tfidf, docs_tfidf)[0]
    indices = sims.argsort()[::-1][:top_k]
    return [(chunks[i], float(sims[i])) for i in indices]


# basic TF-IDF search: given a query, pick the most relevant text chunks
def retrieve_chunks_tfidf(
    chunks: list[dict],
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Retrieve top_k chunks most similar to query using TF-IDF + cosine similarity.
    chunks: list of {chunk_id, text, ...}.
    """
    return [c for c, _ in rank_chunks_tfidf(chunks, query, top_k)]
