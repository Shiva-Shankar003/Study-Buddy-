import sqlite3
from contextlib import contextmanager
import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc TEXT NOT NULL,
    subject TEXT NOT NULL,
    kind TEXT NOT NULL,
    loc TEXT NOT NULL,
    ref TEXT NOT NULL,
    text TEXT NOT NULL,
    parent TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_subject ON chunks (subject);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks (doc);
"""

@contextmanager
def connect():
    con = sqlite3.connect(config.DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()

def init_db():
    with connect() as con:
        con.executescript(SCHEMA)

def doc_exists(doc, subject):
    with connect() as con:
        row = con.execute("SELECT 1 FROM chunks WHERE doc=? AND subject=? LIMIT 1", (doc, subject)).fetchone()
        return row is not None

def add_chunks(rows):
    ids = []
    with connect() as con:
        for r in rows:
            cur = con.execute(
                "INSERT INTO chunks (doc, subject, kind, loc, ref, text, parent) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (r["doc"], r["subject"], r["kind"], str(r["loc"]), r["ref"], r["text"], r["parent"])
            )
            ids.append(cur.lastrowid)
    return ids

def get_chunks(ids):
    ids = list(ids)
    if not ids:
        return []
    marks = ",".join("?" * len(ids))
    with connect() as con:
        rows = con.execute(f"SELECT * FROM chunks WHERE id IN ({marks})", ids).fetchall()
    by_id = {r["id"]: dict(r) for r in rows}
    return [by_id[i] for i in ids if i in by_id]

def all_chunks(subject=None):
    with connect() as con:
        if subject:
            rows = con.execute("SELECT * FROM chunks WHERE subject=?", (subject,)).fetchall()
        else:
            rows = con.execute("SELECT * FROM chunks").fetchall()
        return [dict(r) for r in rows]

def all_chunk_texts(subject=None):
    with connect() as con:
        if subject:
            rows = con.execute("SELECT id, text FROM chunks WHERE subject=?", (subject,)).fetchall()
        else:
            rows = con.execute("SELECT id, text FROM chunks").fetchall()
        return [dict(r) for r in rows]

def list_subjects():
    with connect() as con:
        rows = con.execute("SELECT DISTINCT subject FROM chunks ORDER BY subject").fetchall()
        subs = {r["subject"] for r in rows}
        for table in ("videos", "generated"):
            try:
                rows2 = con.execute(f"SELECT DISTINCT subject FROM {table}").fetchall()
                subs |= {r["subject"] for r in rows2}
            except sqlite3.OperationalError:
                pass
        return sorted(subs)

def list_docs(subject):
    with connect() as con:
        rows = con.execute(
            "SELECT doc, kind, COUNT(*) AS n FROM chunks WHERE subject=? "
            "GROUP BY doc, kind ORDER BY kind, doc", (subject,)).fetchall()
        return [dict(r) for r in rows]

def init_videos():
    with connect() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS videos (
            subject TEXT NOT NULL,
            name TEXT NOT NULL,
            analysis TEXT NOT NULL,
            PRIMARY KEY (subject, name))""")

def save_video(subject, name, analysis):
    with connect() as con:
        con.execute("INSERT OR REPLACE INTO videos (subject, name, analysis) VALUES (?, ?, ?)",
                    (subject, name, analysis))

def list_videos(subject):
    with connect() as con:
        rows = con.execute("SELECT name, analysis FROM videos WHERE subject=? ORDER BY name",
                           (subject,)).fetchall()
        return [dict(r) for r in rows]

def init_generated():
    with connect() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS generated (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT NOT NULL,
            title TEXT NOT NULL,
            kind TEXT NOT NULL,
            content TEXT NOT NULL,
            png BLOB,
            code TEXT,
            source TEXT,
            created TEXT NOT NULL)""")
        con.execute("CREATE INDEX IF NOT EXISTS idx_generated_subject ON generated (subject)")

def save_generated(subject, title, kind, content, png=None, code=None, source=""):
    with connect() as con:
        cur = con.execute(
            "INSERT INTO generated (subject, title, kind, content, png, code, source, created) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now', 'localtime'))",
            (subject, title, kind, content, png, code, source))
        return cur.lastrowid

def list_generated(subject):
    with connect() as con:
        rows = con.execute("SELECT * FROM generated WHERE subject=? ORDER BY id DESC",
                           (subject,)).fetchall()
        return [dict(r) for r in rows]

def delete_generated(item_id):
    with connect() as con:
        con.execute("DELETE FROM generated WHERE id=?", (item_id,))

def delete_doc(subject, doc):
    with connect() as con:
        ids = [r["id"] for r in con.execute(
            "SELECT id FROM chunks WHERE subject=? AND doc=?", (subject, doc)).fetchall()]
        con.execute("DELETE FROM chunks WHERE subject=? AND doc=?", (subject, doc))
        return ids

def delete_video(subject, name):
    with connect() as con:
        con.execute("DELETE FROM videos WHERE subject=? AND name=?", (subject, name))

def delete_subject(subject):
    with connect() as con:
        docs = [dict(r) for r in con.execute(
            "SELECT DISTINCT doc, kind FROM chunks WHERE subject=?", (subject,)).fetchall()]
        ids = [r["id"] for r in con.execute(
            "SELECT id FROM chunks WHERE subject=?", (subject,)).fetchall()]
        vids = [r["name"] for r in con.execute(
            "SELECT name FROM videos WHERE subject=?", (subject,)).fetchall()]
        con.execute("DELETE FROM chunks WHERE subject=?", (subject,))
        con.execute("DELETE FROM videos WHERE subject=?", (subject,))
        con.execute("DELETE FROM generated WHERE subject=?", (subject,))
    return {"chunk_ids": ids, "docs": docs, "videos": vids}

def name_used_elsewhere(name, subject):
    with connect() as con:
        a = con.execute("SELECT 1 FROM chunks WHERE doc=? AND subject!=? LIMIT 1",
                        (name, subject)).fetchone()
        b = con.execute("SELECT 1 FROM videos WHERE name=? AND subject!=? LIMIT 1",
                        (name, subject)).fetchone()
        return a is not None or b is not None

init_db()
init_videos()
init_generated()
