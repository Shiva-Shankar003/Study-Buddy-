"""Retrieval over the indexed chunks (database + Qdrant), with BM25 keyword search.

retrieve(query, top_k=5, subject=None, mode="hybrid", use_rerank=False, **kwargs)
 - query: a string or a list of strings
 - mode: "dense" (meaning search), "bm25" (keyword search) or "hybrid" (both)
Returns a list of chunk dicts (same keys that db.all_chunks gives, plus "score").
"""
import re

from rank_bm25 import BM25Okapi

import db

try:
    import vectorstore
except Exception:  # vectorstore not usable -> keyword search only
    vectorstore = None

_cache = {}


def tokenize(text):
    return re.findall(r"\w+", str(text).lower())


def _load(subject):
    """Chunks of one subject (or all), with a BM25 index. Cached until the chunk count changes."""
    chunks = db.all_chunks(subject) if subject else db.all_chunks()
    key = subject or "*"
    cached = _cache.get(key)
    if cached and cached["n"] == len(chunks):
        return cached
    docs = [tokenize(c.get("text", "")) for c in chunks]
    bm25 = BM25Okapi([d if d else ["_"] for d in docs]) if chunks else None
    cached = {"n": len(chunks), "chunks": chunks, "bm25": bm25}
    _cache[key] = cached
    return cached


def _bm25_rank(query, data, k):
    q = tokenize(query)
    if not q or data["bm25"] is None:
        return []
    scores = data["bm25"].get_scores(q)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [(data["chunks"][i]["id"], float(scores[i])) for i in order[:k] if scores[i] > 0]


def _dense_rank(query, subject, k):
    if vectorstore is None:
        return []
    try:
        return vectorstore.dense_search(query, k=k, subject=subject)
    except Exception as e:
        print(f"[retrieval] dense search failed, using keyword search only: {e}")
        return []


def retrieve(query, top_k=5, subject=None, mode="hybrid", use_rerank=False, **kwargs):
    queries = [query] if isinstance(query, str) else [q for q in query if q]
    if not queries:
        return []
    data = _load(subject)
    if not data["chunks"]:
        return []
    by_id = {c["id"]: c for c in data["chunks"]}

    fused = {}  # reciprocal rank fusion across all rankings
    for q in queries:
        rankings = []
        if mode in ("dense", "hybrid"):
            rankings.append(_dense_rank(q, subject, 20))
        if mode in ("bm25", "hybrid") or not rankings or not rankings[0]:
            rankings.append(_bm25_rank(q, data, 20))
        for ranking in rankings:
            for rank, (cid, _score) in enumerate(ranking, start=1):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (60 + rank)

    results = []
    for cid, score in sorted(fused.items(), key=lambda kv: kv[1], reverse=True):
        if cid in by_id:
            item = dict(by_id[cid])
            item["score"] = float(score)
            results.append(item)
        if len(results) >= top_k:
            break
    return results


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]).strip() or "What are the main topics in the notes?"
    found = retrieve(q)
    if not found:
        print("No relevant notes found.")
    for i, c in enumerate(found, 1):
        print(f"\n[{i}] {c.get('ref', c.get('doc'))}  score={c['score']:.4f}")
        print(str(c.get("text", ""))[:500])