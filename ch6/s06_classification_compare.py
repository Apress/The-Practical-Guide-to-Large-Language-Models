# Import necessary libraries
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, pipeline,
)
import evaluate
import torch

# Dataset: https://huggingface.co/datasets/fancyzhx/ag_news
DATASET = "fancyzhx/ag_news"

# Pre-trained model (base for zero-shot classification)
BASE_MODEL = "roberta-large-mnli"

# Path to the fine-tuned model
TUNED_MODEL_DIR = "/tmp/roberta-large-mnli-agnews-finetuned"

# Test dataset size
TEST_DATASET_SIZE = 100

# Loading the dataset
ds = load_dataset(DATASET)

# ['World','Sports','Business','Sci/Tech']
label_names = ds["train"].features["label"].names

# Loading pre-trained zero-shot classification pipeline
zs = pipeline(
    "zero-shot-classification",
    model = BASE_MODEL,
    tokenizer = BASE_MODEL
)


# Humanizing labels for better readability
def humanize(lbl: str):
    alias = {
        "Sci/Tech": "science and technology",
        "World":    "world",
        "Sports":   "sports",
        "Business": "business",
    }
    return alias.get(lbl, lbl.replace("_", " "))


# Human-readable label names
candidate_labels_h = [humanize(x) for x in label_names]

# Extracting the test subset for evaluation
test_subset = ds["test"].select(range(TEST_DATASET_SIZE))
texts = test_subset["text"]

# Zero-shot classification predictions (Answers)
y_true = test_subset["label"]


# Zero-shot classification predictions function
def zshot_predict(texts, labels_h):
    preds = []
    for t in texts:
        out = zs(
            t,
            labels_h,
            multi_label = False,
            hypothesis_template = "This text is about {}."
        )
        pred_h = out["labels"][0]
        idx = labels_h.index(pred_h)
        preds.append(idx)
    return preds


# Zero-shot classification predictions: Pre-trained model
y_pred_pt = zshot_predict(texts, candidate_labels_h)

# Loading the fine-tuned tokenizer and model
tok = AutoTokenizer.from_pretrained(TUNED_MODEL_DIR)
mdl = AutoModelForSequenceClassification.from_pretrained(TUNED_MODEL_DIR).eval()


# function for predicting indices of labels
def predict_indices(texts, batch_size = 32, max_len = 256):
    preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(
            batch,
            truncation = True,
            padding = True,
            max_length = max_len,
            return_tensors = "pt"
        )
        with torch.no_grad():
            logits = mdl(**enc).logits
        preds.extend(logits.argmax(dim = -1).cpu().tolist())
    return preds


# Fine-tuned model predictions
y_pred_ft = predict_indices(texts)


# Rounding function for better readability
def r(x):
    return round(float(x), 4)


acc = evaluate.load("accuracy")
f1 = evaluate.load("f1")

# Pre-trained model accuracy and F1 score
pt_acc = acc.compute(
    predictions = y_pred_pt,
    references = y_true)["accuracy"]

pt_f1 = f1.compute(
    predictions = y_pred_pt,
    references = y_true,
    average = "macro")["f1"]

# Fine-tuned model accuracy and F1 score
ft_acc = acc.compute(
    predictions = y_pred_ft,
    references = y_true)["accuracy"]

ft_f1 = f1.compute(
    predictions = y_pred_ft,
    references = y_true,
    average = "macro")["f1"]

# Printing the results
print("Overall results on the test subset.")
print(f"Pre-trained model accuracy: {r(pt_acc)}")
print(f"Pre-trained model macro F1: {r(pt_f1)}")
print(f"Fine-tuned model accuracy: {r(ft_acc)}")
print(f"Fine-tuned model macro F1: {r(ft_f1)}")

# Pre-trained model accuracy: 0.68
# Pre-trained model macro F1: 0.6691
# Fine-tuned model accuracy: 0.93
# Fine-tuned model macro F1: 0.9168
