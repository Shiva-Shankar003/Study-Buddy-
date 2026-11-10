import json
import random

import db
from llm import ask_json

# Test documents (from eval_data.jsonl) are NEVER used for fine-tuning: this prevents leakage.
test_docs = {json.loads(line)["doc"] for line in open("eval_data.jsonl", encoding="utf-8")
             if line.strip()}
pool = [c for c in db.all_chunks() if c["doc"] not in test_docs and len(c["text"]) > 200]
random.seed(0)
random.shuffle(pool)

pairs = 0
with open("train_pairs.jsonl", "w", encoding="utf-8") as out:
    for c in pool[:300]:
        prompt = ("Write 2 different questions a student could ask whose answer is in this "
                  "passage. Return a JSON list of strings only.\n\nPassage:\n" + c["text"])
        for q in ask_json(prompt) or []:
            if isinstance(q, str) and len(q) > 10:
                out.write(json.dumps({"question": q, "passage": c["text"]}) + "\n")
                pairs += 1
print("wrote", pairs, "training pairs from", len(pool[:300]), "chunks")
