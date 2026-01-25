"""
TF-IDF retrieval over chunk texts. Top-k chunks for a query.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def retrieve_chunks_tfidf(
    chunks: list[dict],
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Retrieve top_k chunks most similar to query using TF-IDF + cosine similarity.
    chunks: list of {chunk_id, text, ...}.
    """
    if not chunks:
        return []
    if not (query and query.strip()):
        return chunks[:top_k]
    query = query.strip()
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer()
    docs_tfidf = vectorizer.fit_transform(texts)
    query_tfidf = vectorizer.transform([query])
    sims = cosine_similarity(query_tfidf, docs_tfidf)[0]
    indices = sims.argsort()[::-1][:top_k]
    return [chunks[i] for i in indices]
