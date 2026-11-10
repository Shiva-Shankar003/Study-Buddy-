import atexit
import numpy as np
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import config

_client = None
_embedders = {}

def get_client():
    global _client
    if _client is None:
        if config.QDRANT_URL:
            _client = QdrantClient(url=config.QDRANT_URL)
        else:
            _client = QdrantClient(path=config.QDRANT_PATH)
        atexit.register(_client.close)
    return _client

def get_embedder(name=None):
    name = name or config.EMBED_MODEL
    if name not in _embedders:
        _embedders[name] = SentenceTransformer(name)
    return _embedders[name]

def embed(texts, model=None):
    vecs = get_embedder(model).encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype="float32")

def ensure_collection(collection=None, model=None):
    name = collection or config.COLLECTION
    client = get_client()
    if not client.collection_exists(name):
        dim = int(embed(["dimension probe"], model).shape[1])
        params = models.VectorParams(size=dim, distance=models.Distance.COSINE)
        client.create_collection(collection_name=name, vectors_config=params)
    return name

def add_vectors(ids, texts, subjects, collection=None, model=None, batch=256):
    name = ensure_collection(collection, model)
    for s in range(0, len(ids), batch):
        vecs = embed(texts[s:s+batch], model)
        points = [
            models.PointStruct(id=int(i), vector=v.tolist(), payload={"subject": sub})
            for i, v, sub in zip(ids[s:s+batch], vecs, subjects[s:s+batch])
        ]
        get_client().upsert(collection_name=name, points=points)

def dense_search(query, k=20, collection=None, model=None, subject=None):
    name = collection or config.COLLECTION
    client = get_client()
    if not client.collection_exists(name):
        return []
    flt = None
    if subject:
        flt = models.Filter(must=[models.FieldCondition(key="subject", match=models.MatchValue(value=subject))])
    vec = embed([query], model)[0].tolist()
    res = client.query_points(collection_name=name, query=vec, limit=k, query_filter=flt)
    return [(int(p.id), float(p.score)) for p in res.points]