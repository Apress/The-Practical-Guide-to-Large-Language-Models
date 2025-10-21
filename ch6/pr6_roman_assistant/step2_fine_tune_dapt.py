import os

# If you want to run on CPU, set this flag to False
USE_CUDA = True

if not USE_CUDA:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Importing necessary libraries
import torch
from itertools import chain
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments,
)

# Pre-trained model
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Path to the dataset (relative to the script location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_GLOB = os.path.join(SCRIPT_DIR, "data", "roman_empire_txt", "*.txt")

# Output directory for the adapted (trained) model
TUNED_MODEL = "/tmp/roman_assistant_model"

# Loading the text dataset
dataset = load_dataset(
    "text",
    data_files = {"train": DATA_GLOB},
    split = "train"
)

# Loading the tokenizer and model
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast = True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# If the tokenizer does not have a pad token, set it to the eos token
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# For training, disable the cache
model.config.use_cache = False

# Resize token embeddings to match the tokenizer's vocabulary size
model.resize_token_embeddings(len(tok))

# Setting the device (GPU or CPU)
device = torch.device("cuda")\
    if USE_CUDA and torch.cuda.is_available()\
    else torch.device("cpu")

# Moving the model to the selected device
model.to(device)

# The length of text blocks for training.
# Text will be processed in chunks of this size.
# It can be reduced to save memory, but too small values may hurt quality.
SEQ_LEN = 1024


# Tokenization function
def tokenize_fn(batch):
    return tok(
        batch["text"],
        add_special_tokens = False,
        return_attention_mask = False
    )


# Tokenized dataset
tokenized = dataset.map(
    tokenize_fn,
    batched = True,
    remove_columns = dataset.column_names
)


# Function to group texts into fixed-size blocks
def group_texts(examples):
    # Concatenate all texts and split into blocks of SEQ_LEN
    concatenated = list(chain(*examples["input_ids"]))
    total_length = (len(concatenated) // SEQ_LEN) * SEQ_LEN
    concatenated = concatenated[:total_length]
    input_blocks = [concatenated[i:i + SEQ_LEN] for i in range(0, total_length, SEQ_LEN)]
    return {"input_ids": input_blocks, "labels": [blk[:] for blk in input_blocks]}


# Dataset for language modeling
lm_dataset = tokenized.map(
    group_texts,
    batched = True,
    remove_columns = tokenized.column_names,
    desc = "Packing tokens into fixed-size blocks"
)

# Data collator for language modeling task
collator = DataCollatorForLanguageModeling(tok, mlm = False)

# Training arguments
training_args = TrainingArguments(
    output_dir = TUNED_MODEL,
    num_train_epochs = 10,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1,
    learning_rate = 2e-5,
    weight_decay = 0.01,
    warmup_ratio = 0.03,
    logging_steps = 10,
    # save_strategy = "epoch",
    save_strategy = "no",
    dataloader_num_workers = 1,
    use_cpu = (not USE_CUDA),
    fp16 = (USE_CUDA and torch.cuda.is_available()),
    bf16 = False,
    eval_strategy = "no",
    # Reporting settings
    report_to = ["tensorboard"],
    logging_dir = os.path.join(TUNED_MODEL, "logs"),
)

# Initializing the Trainer
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = lm_dataset,
    processing_class = tok,
    data_collator = collator,
)

# Instructions for running TensorBoard
print("========================================")
print("Run tensorboard to monitor training:")
print(f"tensorboard --logdir {os.path.join(TUNED_MODEL, 'logs')} --bind_all")
print("========================================")

# Training the model
print("Starting DAPT training...")
trainer.train()
print("Training completed successfully!")

# Saving the fine-tuned model and tokenizer
trainer.save_model(TUNED_MODEL)
tok.save_pretrained(TUNED_MODEL)
print(f"Model and tokenizer saved to {TUNED_MODEL}")
