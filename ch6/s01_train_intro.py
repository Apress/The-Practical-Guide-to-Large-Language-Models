import os

# Define whether to use CUDA or not for training
# You can set this to False if you want to train on CPU or if CUDA is not available
USE_CUDA = True
if not USE_CUDA:
    # If CUDA is not used, hide the GPU from the process
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Import necessary libraries
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments,
)

# Pre-trained model
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Path to the training dataset: alien_romulus_qa.jsonl
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "alien_romulus_qa.jsonl")

# Output directory for the fine-tuned model
# Please replace this path with your own if needed
# This is the directory where the fine-tuned model and tokenizer will be saved
TUNED_MODEL_DIR = "/tmp/qwen-alien_romulus-finetuned"

# Loading and formatting the dataset from JSONL file
dataset = load_dataset("json", data_files = DATA_PATH, split = "train")


# Formatting the dataset samples to question-answer pairs
def format_sample(sample):
    return f"Question: {sample['instruction']}\nAnswer: {sample['output']}"


# Applying the formatting function to the dataset
dataset = dataset.map(lambda s: {"text": format_sample(s)})

# Tokenizer initialization
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

# Setting the pad token if it is not defined
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


# Tokenization function
def tokenize_fn(samples):
    return tok(
        samples["text"],
        truncation = True,
        padding = "max_length",
        max_length = 256
    )


# Tokenizing the dataset
tokenized = dataset.map(
    tokenize_fn,
    batched = True,
    remove_columns = dataset.column_names
)

# Initializing the pre-trained model
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Resizing the token embeddings to match the tokenizer vocabulary size
model.resize_token_embeddings(len(tok))

# Moving the model to device: CUDA or CPU
device = torch.device("cuda")\
    if (USE_CUDA and torch.cuda.is_available())\
    else torch.device("cpu")

model.to(device)

# Disabling the cache to avoid memory issues during training
model.config.use_cache = False

# Data collator for language modeling task
collator = DataCollatorForLanguageModeling(tok, mlm = False)

# Training arguments for the Trainer
training_args = TrainingArguments(
    # Output directory for the fine-tuned model
    output_dir = TUNED_MODEL_DIR,
    # Number of training epochs
    num_train_epochs = 5,
    # Batch size and learning rate settings
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1,
    learning_rate = 2e-5,
    weight_decay = 0.01,
    warmup_ratio = 0.03,
    logging_steps = 10,
    # Save strategy and reporting settings
    save_strategy = "no",
    report_to = "none",
    # Device settings
    no_cuda = (not USE_CUDA),
    fp16 = (USE_CUDA and torch.cuda.is_available()),
    bf16 = False,
    # Number of workers for data loading
    dataloader_num_workers = 1,
)

# Trainer initialization
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = tokenized,
    processing_class = tok,
    data_collator = collator,
)

# Training the model
print("Starting training...")
trainer.train()
# {'loss': 0.1658, 'grad_norm': 14.325278282165527, 'learning_rate': 1.6877637130801689e-06, 'epoch': 4.69}
# {'loss': 0.2537, 'grad_norm': 23.56131362915039, 'learning_rate': 8.438818565400844e-07, 'epoch': 4.9}
# {'train_runtime': 60.4376, 'train_samples_per_second': 4.054, 'train_steps_per_second': 4.054, 'train_loss':
# 0.7917323005442717, 'epoch': 5.0}
print("Training completed successfully!")

# Saving the fine-tuned model and tokenizer
trainer.save_model(TUNED_MODEL_DIR)
tok.save_pretrained(TUNED_MODEL_DIR)
print(f"Model and tokenizer saved to {TUNED_MODEL_DIR}")
