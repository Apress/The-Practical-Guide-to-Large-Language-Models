# Importing necessary libraries
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
import torch

# Number of samples to evaluate
n_samples = 1000
# Pre-trained model
model_id = "distilbert-base-uncased-finetuned-sst-2-english"

# Loading the Yelp Polarity dataset for testing
ds = load_dataset("fancyzhx/yelp_polarity", split = "test")

# Random subset
ds = ds.shuffle().select(range(n_samples))
texts = ds["text"]
texts = [t if isinstance(t, str) else "" if t is None else str(t) for t in texts]

# Loading model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

# Choosing device: GPU if available, else CPU
device = 0 if torch.cuda.is_available() else -1
if device == 0:
    print("Device set to use cuda:0")

# Creating a text classification pipeline
clf = pipeline(
    task="text-classification",
    model = model,
    tokenizer = tokenizer,
    device = device,
    truncation=True
)

# Getting predictions
preds = clf(
    texts,
    batch_size=32,
    padding=True,
)

# Calculating accuracy
correct = 0
for p, y in zip(preds, ds["label"]):
    label = p["label"]
    # POSITIVE/NEGATIVE -> 1/0
    pred_int = 1 if label.upper().startswith("POS") else 0
    correct += int(pred_int == y)

# Accuracy
acc = correct / len(ds)

print(f"Accuracy on {n_samples} random samples: {acc:.2%}")
# 90.30%
