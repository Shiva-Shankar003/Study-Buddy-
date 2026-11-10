import argparse

import db
from vectorstore import add_vectors

ap = argparse.ArgumentParser(description="Re-embed every stored chunk into a new collection")
ap.add_argument("--embed-model", required=True)
ap.add_argument("--collection", required=True)
a = ap.parse_args()

chunks = db.all_chunks()
add_vectors([c["id"] for c in chunks], [c["text"] for c in chunks],
            [c["subject"] for c in chunks], collection=a.collection, model=a.embed_model)
print("indexed", len(chunks), "chunks into", a.collection)
