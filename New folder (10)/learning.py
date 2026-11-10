import time
import db

SCHEMA = """
CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, subject TEXT, topic TEXT, correct INTEGER
);
CREATE TABLE IF NOT EXISTS cards (
    id INTEGER PRIMARY KEY AUTOINCREMENT, subject TEXT, topic TEXT, front TEXT, back TEXT,
    reps INTEGER DEFAULT 0, interval INTEGER DEFAULT 0, ef REAL DEFAULT 2.5, due REAL
);
"""
with db.connect() as _con:
    _con.executescript(SCHEMA)

HALF_LIFE_DAYS = 14.0  # an answer from 14 days ago counts half as much as one from today


def record_attempt(subject, topic, correct):
    with db.connect() as con:
        con.execute("INSERT INTO attempts (ts, subject, topic, correct) VALUES (?, ?, ?, ?)",
                    (time.time(), subject, topic, int(correct)))


def mastery(subject):
    """Recency-weighted accuracy per topic, between 0 and 1 (with a small prior of 0.5)."""
    now = time.time()
    with db.connect() as con:
        rows = con.execute("SELECT ts, topic, correct FROM attempts WHERE subject=?",
                           (subject,)).fetchall()
    num, den = {}, {}
    for r in rows:
        weight = 0.5 ** (((now - r["ts"]) / 86400) / HALF_LIFE_DAYS)
        num[r["topic"]] = num.get(r["topic"], 0.0) + weight * r["correct"]
        den[r["topic"]] = den.get(r["topic"], 0.0) + weight
    return {t: (num[t] + 0.5) / (den[t] + 1.0) for t in num}


def weakest_topics(subject, n=3, below=0.7):
    scores = mastery(subject)
    ranked = sorted((v, t) for t, v in scores.items() if v < below)
    return [t for _, t in ranked[:n]]


def level_for(avg_mastery):
    """Pick the Bloom level of new quiz questions from the average mastery."""
    if avg_mastery < 0.4:
        return "remember"
    if avg_mastery < 0.7:
        return "understand"
    return "apply"


def sm2(quality, reps, interval, ef):
    """SM-2 spaced repetition. quality: 0 (blackout) to 5 (perfect recall)."""
    if quality < 3:
        reps, interval = 0, 1
    else:
        interval = 1 if reps == 0 else 6 if reps == 1 else round(interval * ef)
        reps += 1
    ef = max(1.3, ef + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    return reps, interval, ef


def add_card(subject, topic, front, back):
    with db.connect() as con:
        con.execute("INSERT INTO cards (subject, topic, front, back, due) VALUES (?, ?, ?, ?, ?)",
                    (subject, topic, front, back, time.time()))


def due_cards(limit=20):
    with db.connect() as con:
        rows = con.execute("SELECT * FROM cards WHERE due <= ? ORDER BY due LIMIT ?",
                           (time.time(), limit)).fetchall()
    return [dict(r) for r in rows]


def review_card(card_id, quality):
    with db.connect() as con:
        c = con.execute("SELECT reps, interval, ef FROM cards WHERE id=?", (card_id,)).fetchone()
        reps, interval, ef = sm2(quality, c["reps"], c["interval"], c["ef"])
        con.execute("UPDATE cards SET reps=?, interval=?, ef=?, due=? WHERE id=?",
                    (reps, interval, ef, time.time() + interval * 86400, card_id))
