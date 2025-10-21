import os

# Define whether to use CUDA or not for training
# You can set this to False if you want to train on CPU or if CUDA is not available
USE_CUDA = True

if not USE_CUDA:
    # If CUDA is not used, hide the GPU from the process
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Import necessary libraries
# pip install evaluate==0.4.3
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer,
)
import numpy as np
import evaluate
import torch

# Define the dataset for model fine-tuning
# Dataset: https://huggingface.co/datasets/fancyzhx/ag_news
DATASET = "fancyzhx/ag_news"

# Pre-trained model
BASE_MODEL = "roberta-large-mnli"

# Output directory for the fine-tuned model
# Please replace this path with your own if needed
# This is the directory where the fine-tuned model and tokenizer will be saved
TUNED_MODEL_DIR = "/tmp/roberta-large-mnli-agnews-finetuned"

# Maximum sequence length for tokenization.
# Text longer than this will be truncated
MAX_LEN = 128

# Size of the training set for fine-tuning
# You can increase or decrease this value based on your needs and resources
TRAIN_MODEL_SIZE = 10_000

# Loading the dataset
ds = load_dataset(DATASET)

# Extracting label names and creating mappings
# Labels: ['World','Sports','Business','Sci/Tech']
label_names = ds["train"].features["label"].names
id2label = {i: n for i, n in enumerate(label_names)}
label2id = {n: i for i, n in enumerate(label_names)}

# Cut down the training set for faster fine-tuning
if "train" in ds and len(ds["train"]) > TRAIN_MODEL_SIZE:
    ds["train"] = ds["train"].shuffle().select(range(TRAIN_MODEL_SIZE))

# Creating validation set if it does not exist
if "validation" not in ds:
    ds_split = ds["train"].train_test_split(test_size = 0.3)
    ds["validation"] = ds_split["test"]
    ds["train"] = ds_split["train"]

# Loading evaluation metrics
# accuracy: measures the proportion of correct predictions
acc = evaluate.load("accuracy")
# f1: harmonic mean of precision and recall, useful for imbalanced datasets
f1 = evaluate.load("f1")

# Tokenizer initialization
tok = AutoTokenizer.from_pretrained(BASE_MODEL)


# Tokenization function
def tokenize(batch):
    enc = tok(
        batch["text"],
        truncation = True,
        max_length = MAX_LEN
    )
    enc["labels"] = batch["label"]
    return enc


# Tokenizing train dataset
train_tok = ds["train"].map(
    tokenize,
    batched = True,
    remove_columns = ["text", "label"]
)

# Tokenizing validation dataset
val_tok = ds["validation"].map(
    tokenize,
    batched = True,
    remove_columns = ["text", "label"]
)

# Tokenizing test dataset
test_tok = ds["test"].map(
    tokenize,
    batched = True,
    remove_columns = ["text", "label"]
)

# Initializing the pre-trained model for fine-tuning
ft_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels = len(label_names),
    id2label = id2label,
    label2id = label2id,
    problem_type = "single_label_classification",
    ignore_mismatched_sizes = True
)

# Moving the model to device: CUDA or CPU
device = torch.device("cuda")\
    if (USE_CUDA and torch.cuda.is_available())\
    else torch.device("cpu")

ft_model.to(device)

# Disable the cache for training
ft_model.config.use_cache = False


# Function to compute metrics during evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis = -1)
    return {
        "accuracy": acc.compute(
            predictions = preds,
            references = labels)["accuracy"],
        "macro_f1": f1.compute(
            predictions = preds,
            references = labels,
            average = "macro")["f1"],
    }


# Defining training arguments
args = TrainingArguments(
    # Output directory for the fine-tuned model
    output_dir = TUNED_MODEL_DIR,
    # Batch size for training and evaluation
    per_device_train_batch_size = 8 if USE_CUDA else 1,
    per_device_eval_batch_size = 32 if USE_CUDA else 1,
    # Number of gradient accumulation steps
    gradient_accumulation_steps = 1 if USE_CUDA else 4,
    # Data loader settings
    dataloader_num_workers = 0 if not USE_CUDA else 2,
    # Learning rate and weight decay settings
    learning_rate = 2e-5,
    weight_decay = 0.01,
    warmup_ratio = 0.06,
    # Number of training epochs
    num_train_epochs = 5,
    # Evaluation strategy: evaluate after each epoch
    eval_strategy = "epoch",
    # Save strategy: save the model after each epoch
    save_strategy = "epoch",
    # Device settings
    no_cuda = (not USE_CUDA),
    fp16 = (USE_CUDA and torch.cuda.is_available()),
    bf16 = False,
    # Logging settings
    logging_steps = 50,
    # Reporting settings
    report_to = ["tensorboard"],
    logging_dir = os.path.join(TUNED_MODEL_DIR, "logs"),
)

# Initializing the Trainer
trainer = Trainer(
    model = ft_model,
    args = args,
    train_dataset = train_tok,
    eval_dataset = val_tok,
    processing_class = tok,
    compute_metrics = compute_metrics,
)

# Here we can run tensorboard to monitor training
print('========================================')
print("Run tensorboard to monitor training:")
# Run: $ tensorboard --logdir ${TUNED_MODEL_DIR} --bind_all
print(f"tensorboard --logdir {os.path.join(TUNED_MODEL_DIR, 'logs')} --bind_all")
print('========================================')

# Training the model
trainer.train()

# Saving the fine-tuned model
trainer.save_model(TUNED_MODEL_DIR)

# Saving the tokenizer
tok.save_pretrained(TUNED_MODEL_DIR)
