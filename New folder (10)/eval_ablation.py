import json
import os
import sys

import pandas as pd

from agent import DEFAULTS, plain_answer, run
from verify import verify_answer

# Judge: use a DIFFERENT provider from the one that writes answers when you can
# (for example SB_JUDGE_PROVIDER=groq while SB_LLM_PROVIDER=gemini). Same provider = biased scores.
JUDGE = os.getenv("SB_JUDGE_PROVIDER") or None

OFF = dict(use_rewrite=False, use_hyde=False, use_rerank=False, use_crag=False, use_verify=False)
CONFIGS = {
    "plain_llm": None,
    "naive_rag": {**OFF, "retrieval": "dense"},
    "+hybrid": {**OFF, "retrieval": "hybrid"},
    "+reranker": {**OFF, "retrieval": "hybrid", "use_rerank": True},
    "+rewrite_hyde": {**OFF, "retrieval": "hybrid", "use_rerank": True,
                      "use_rewrite": True, "use_hyde": True},
    "+crag_verify": {"retrieval": "hybrid", "use_rerank": True, "use_rewrite": True,
                     "use_hyde": True, "use_crag": True, "use_verify": True},
    "+finetuned_embeddings": {"retrieval": "hybrid", "use_rerank": True, "use_rewrite": True,
                              "use_hyde": True, "use_crag": True, "use_verify": True,
                              "embed_model": "models/bge-ft", "collection": "sb_ft"},
}


def is_hit(chunk, gold):
    """A retrieved chunk is correct if it comes from the gold document and gold page / time."""
    if chunk["doc"] != gold["doc"]:
        return False
    if chunk["kind"] == "youtube":
        return abs(int(chunk["loc"]) - int(gold["loc"])) <= 60
    return str(chunk["loc"]) == str(gold["loc"])


def score_one(item, out):
    chunks = out.get("chunks", [])
    rank = next((i for i, c in enumerate(chunks, start=1) if is_hit(c, item)), None)
    row = {"hit@5": float(rank is not None and rank <= 5), "mrr": 1.0 / rank if rank else 0.0,
           "latency_s": out["latency"], "faithfulness": None, "citation_precision": None}
    if chunks and out.get("answer"):  # judge the FINAL answer for every RAG config
        v = verify_answer(out["answer"], chunks, provider=JUDGE)
        row["faithfulness"], row["citation_precision"] = v["faithfulness"], v["citation_precision"]
    return row


def main(path="eval_data.jsonl"):
    items = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    table = []
    for name, overrides in CONFIGS.items():
        rows = []
        for item in items:
            if overrides is None:
                out = plain_answer(item["question"])
            else:
                out = run(item["question"], item.get("subject"), **overrides)
            rows.append(score_one(item, out))
        df = pd.DataFrame(rows)
        table.append({"config": name, **df.mean(numeric_only=True).to_dict()})
        print("finished", name, flush=True)
    result = pd.DataFrame(table).round(3)
    result.to_csv("ablation_results.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eval_data.jsonl")
