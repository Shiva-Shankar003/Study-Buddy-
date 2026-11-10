import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall,
                           ResponseRelevancy)

import config
from agent import run

# Written for RAGAS 0.2.x. If an import fails, check the RAGAS docs for your installed version.
items = [json.loads(line) for line in open("eval_data.jsonl", encoding="utf-8") if line.strip()]
samples = []
for item in items:
    out = run(item["question"], item.get("subject"))
    samples.append({"user_input": item["question"], "response": out.get("answer", ""),
                    "retrieved_contexts": [c["parent"] for c in out.get("chunks", [])],
                    "reference": item["answer"]})

llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model=config.GEMINI_MODEL,
                                                 google_api_key=config.GEMINI_API_KEY))
emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=config.EMBED_MODEL))
result = evaluate(EvaluationDataset.from_list(samples),
                  metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(),
                           LLMContextRecall()], llm=llm, embeddings=emb)
print(result)
result.to_pandas().to_csv("ragas_results.csv", index=False)
