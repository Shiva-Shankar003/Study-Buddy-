import random
import db
import learning
from llm import ask_json
from retrieval import retrieve


def _pick_sources(subject, n):
    """Weak topics first (found with search), then random chunks from the subject."""
    sources, seen = [], set()
    for topic in learning.weakest_topics(subject, 2):
        for c in retrieve([topic], subject=subject, mode="hybrid", top_k=2, use_rerank=False):
            if (c["doc"], c["loc"]) not in seen:
                seen.add((c["doc"], c["loc"]))
                sources.append(c)
    pool = db.all_chunks(subject)
    random.shuffle(pool)
    for c in pool:
        if len(sources) >= n:
            break
        if (c["doc"], c["loc"]) not in seen:
            seen.add((c["doc"], c["loc"]))
            sources.append(c)
    return sources[:n]


def make_quiz(subject, n=5, level=None):
    sources = _pick_sources(subject, n)
    if not sources:
        return []
    if level is None:
        scores = learning.mastery(subject)
        level = learning.level_for(sum(scores.values()) / len(scores) if scores else 0.5)
    texts = "\n\n".join(f"Text {i}: {c['parent'][:1500]}" for i, c in enumerate(sources, start=1))
    prompt = (f"Write exactly {len(sources)} multiple-choice questions, one per text, at Bloom "
              f"level '{level}'. Each question must be answerable from its text only. "
              "Return a JSON list only. Each item: "
              '{"text_id": 1, "topic": "2-4 word topic name", "question": "...", '
              '"options": ["...", "...", "...", "..."], "answer_index": 0, '
              '"explanation": "..."}\n\n'
              + texts)
    quiz = []
    for item in ask_json(prompt, cache=False) or []:
        try:
            options, answer = list(item["options"]), int(item["answer_index"])
            if len(options) != 4 or not 0 <= answer < 4:
                continue
            quiz.append({"subject": subject, "topic": str(item["topic"]),
                         "question": str(item["question"]), "options": options,
                         "answer_index": answer, "explanation": str(item["explanation"]),
                         "ref": sources[int(item["text_id"]) - 1]["ref"], "level": level})
        except (KeyError, ValueError, TypeError, IndexError):
            continue
    return quiz


def make_flashcards(subject, n=8):
    """Create n flashcards from random parts of the notes and store them for SM-2 review."""
    pool = db.all_chunks(subject)
    random.shuffle(pool)
    texts = "\n\n".join(c["parent"][:1200] for c in pool[:n])
    if not texts:
        return 0
    prompt = (f"Write {n} flashcards from the notes below. Return a JSON list only. Each item: "
              '{"topic": "2-4 word topic name", "front": "question", "back": "short answer"}\n\n'
              + texts)
    made = 0
    for item in ask_json(prompt, cache=False) or []:
        try:
            learning.add_card(subject, str(item["topic"]), str(item["front"]), str(item["back"]))
            made += 1
        except (KeyError, TypeError):
            continue
    return made
