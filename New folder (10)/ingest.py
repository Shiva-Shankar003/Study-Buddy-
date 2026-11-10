import argparse
import os
import db
from chunker import make_chunks
from ocr import load_handwritten
from pdf_loader import load_pdf
from vectorstore import add_vectors
from yt_loader import load_youtube

def _store(units, doc, subject, kind):
    if db.doc_exists(doc, subject):
        print(f"Skipped (already indexed): {doc}")
        return 0
    rows = make_chunks(units, doc, subject, kind)
    if not rows:
        return 0
    ids = db.add_chunks(rows)
    add_vectors(ids, [r["text"] for r in rows], [subject] * len(rows))
    return len(rows)

def ingest_pdf(path, subject):
    return _store(load_pdf(path), os.path.basename(path), subject, "pdf")

def ingest_image(path, subject, engine=None, page=1):
    units = load_handwritten(path, engine, page)
    return _store(units, os.path.basename(path), subject, "handwritten")

def ingest_youtube(url, subject):
    return _store(load_youtube(url), url, subject, "youtube")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Add study material to StudyBuddy")
    ap.add_argument("--subject", required=True)
    ap.add_argument("--pdf", nargs="*", default=[])
    ap.add_argument("--image", nargs="*", default=[])
    ap.add_argument("--youtube", nargs="*", default=[])
    a = ap.parse_args()

    for p in a.pdf:
        print(p, "->", ingest_pdf(p, a.subject), "chunks")
    for p in a.image:
        print(p, "->", ingest_image(p, a.subject), "chunks")
    for u in a.youtube:
        print(u, "->", ingest_youtube(u, a.subject), "chunks")