import json

from datasets import Dataset
from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments, losses)

import config

rows = [json.loads(line) for line in open("train_pairs.jsonl", encoding="utf-8")]
data = Dataset.from_dict({"anchor": [r["question"] for r in rows],
                          "positive": [r["passage"] for r in rows]})

# On Colab/Kaggle replace config.EMBED_MODEL with "BAAI/bge-small-en-v1.5"
model = SentenceTransformer(config.EMBED_MODEL)
loss = losses.MultipleNegativesRankingLoss(model)  # other passages in the batch act as negatives
args = SentenceTransformerTrainingArguments(
    output_dir="models/bge-ft-checkpoints", num_train_epochs=2, per_device_train_batch_size=16,
    learning_rate=2e-5, warmup_ratio=0.1, logging_steps=10, report_to="none")
SentenceTransformerTrainer(model=model, args=args, train_dataset=data, loss=loss).train()
model.save("models/bge-ft")
print("saved models/bge-ft")
