import os

# Define whether to use CUDA or not
USE_CUDA = True

if not USE_CUDA:
    # If CUDA is not used, hide the GPU from the process
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Import necessary libraries
import torch
from itertools import chain
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments,
)

# Pre-trained model
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Dataset path (relative to the current script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_GLOB = os.path.join(SCRIPT_DIR, "data", "space_missions_corpus", "*.txt")

# Tuned model output directory
TUNED_MODEL = "/tmp/qwen_dapt_model"

# Sequence length for training.
# Text is processed in chunks of this size.
SEQ_LEN = 1024

# Loading the corpus dataset (column 'text')
dataset = load_dataset(
    "text",
    data_files = {"train": DATA_GLOB},
    split = "train"
)

# Initializing the tokenizer and model
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast = True)

# Setting the pad token if it is not defined.
# Pad token means "no token" and is used to fill empty space in sequences
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# Initializing the pre-trained model
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# For training, we disable the cache
model.config.use_cache = False

# Resize the token embeddings to match the tokenizer's vocabulary size
model.resize_token_embeddings(len(tok))

# Setting the device
device = torch.device("cuda")\
    if (USE_CUDA and torch.cuda.is_available())\
    else torch.device("cpu")

# Moving the model to the device
model.to(device)


# Tokenization function
def tokenize_fn(batch):
    # We will add eos_token_id manually to avoid extra bos_token_id at the beginning
    # which is not needed for causal language modeling.
    # Also, it helps to have a single consistent token at the end of each chunk
    # (even if the text already ends with a punctuation mark)
    # This way, the model learns to predict the end of text more reliably.
    ids = []
    for t in batch["text"]:
        ids.append(tok(t, add_special_tokens = False)["input_ids"] + [tok.eos_token_id])
    return {"input_ids": ids}


# Tokenizing the dataset
tokenized = dataset.map(
    tokenize_fn,
    batched = True,
    remove_columns = dataset.column_names
)


# Packing function to group tokens into fixed-size blocks
def group_texts(examples):
    # Concatenate all tokens and split into blocks of SEQ_LEN
    concatenated = list(chain(*examples["input_ids"]))
    total_length = (len(concatenated) // SEQ_LEN) * SEQ_LEN
    concatenated = concatenated[:total_length]
    input_blocks = [concatenated[i:i + SEQ_LEN]
                    for i in range(0, total_length, SEQ_LEN)]
    return {
        "input_ids": input_blocks,
        "labels":    [blk[:] for blk in input_blocks]
    }


# Packing tokens into fixed-size blocks
lm_dataset = tokenized.map(
    group_texts,
    batched = True,
    remove_columns = tokenized.column_names,
    desc = "Packing tokens into fixed-size blocks"
)

# Data collator for language modeling task
# Data collator will dynamically pad the inputs received,
# as well as create the labels tensor.
# Padding example: [1, 2, 3] -> [1, 2, 3, 0, 0] if max length in batch is 5.
collator = DataCollatorForLanguageModeling(tok, mlm = False)

# Training arguments for the Trainer
training_args = TrainingArguments(
    # Output directory for the fine-tuned model
    output_dir = TUNED_MODEL,
    # Number of training epochs
    num_train_epochs = 30,
    # Batch sizes for training and evaluation
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1,
    dataloader_num_workers = 1,
    # Learning rate and other hyperparameters
    learning_rate = 2e-5,
    weight_decay = 0.01,
    warmup_ratio = 0.03,
    logging_steps = 10,
    # Save strategy: no saving during training
    save_strategy = "no",
    eval_strategy = "no",
    # Disabling CUDA if not used
    use_cpu = (not USE_CUDA),
    fp16 = (USE_CUDA and torch.cuda.is_available()),
    bf16 = False,
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
