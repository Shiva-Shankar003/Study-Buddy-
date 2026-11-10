from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

import db
import learning
from agent import run

app = FastAPI(title="StudyBuddy API")


class AskBody(BaseModel):
    question: str
    subject: Optional[str] = None


@app.get("/subjects")
def subjects():
    return db.list_subjects()


@app.post("/ask")
def ask_endpoint(body: AskBody):
    out = run(body.question, body.subject)
    v = out.get("verification") or {}
    return {
        "intent": out["intent"],
        "answer": out.get("answer", ""),
        "sources": [{"n": i, "ref": c["ref"]} for i, c in enumerate(out.get("chunks", []), 1)],
        "faithfulness": v.get("faithfulness"),
        "citation_precision": v.get("citation_precision"),
        "latency_s": round(out["latency"], 2),
    }


@app.get("/mastery/{subject}")
def mastery(subject: str):
    return learning.mastery(subject)
