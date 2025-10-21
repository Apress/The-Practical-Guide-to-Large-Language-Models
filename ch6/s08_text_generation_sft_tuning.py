import os

# Define whether to use CUDA or not
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

# Pre-trained model (base for fine-tuning)
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Path to the training dataset
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "smartphone_qa.jsonl")

# Path to the fine-tuned model output directory
TUNED_MODEL = "/tmp/qwen_smartphone-qa-finetuned"

# MAX_LEN is the maximum sequence length for training.
MAX_LEN = 256

# Loading and formatting the dataset from JSONL file
dataset = load_dataset(
    "json",
    data_files = DATA_PATH,
    split = "train"
)


# Build separate prompt and answer fields.
# We keep the "Answer:" prefix in the prompt so the model knows where the answer starts,
# but the loss will be computed ONLY on the answer tokens.
def build_example(s):
    prompt = f"Question: {s['instruction']}\nAnswer:"
    # A leading space before the answer helps many tokenizers produce cleaner tokens
    # We also append EOS to teach the model when to stop.
    answer = f" {s['response']}"
    return {"prompt": prompt, "answer": answer}


# Creating a new column 'text' with formatted samples
# Example: {
# "text": {
#   "prompt": "Question: ...\nAnswer:",
#   "answer": " ..."
#   }
# }
dataset = dataset.map(lambda s: {"text": build_example(s)})

# Tokenizer initialization
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

# Setting the pad token if it is not defined
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


# Tokenize with explicit labels masking.
# We create input_ids, attention_mask and labels where:
# - labels are -100 for everything BEFORE the answer (no loss on the prompt),
# - labels are -100 on padding,
# - labels equal input_ids for the answer span (loss computed only there).
def tokenize_with_labels(example):
    # Encode prompt alone to get its length in tokens
    # We will use this to mask out the prompt in labels to avoid computing loss on it
    prompt_ids = tok(
        example["instruction"],
        add_special_tokens = True,
        return_attention_mask = False
    )["input_ids"]

    # Encode full sequence (prompt + answer + EOS)
    full_text = example["instruction"] + example["response"] + tok.eos_token
    encoded = tok(
        full_text,
        truncation = True,
        # use fixed max length for simplicity; dynamic padding also works
        padding = "max_length",
        max_length = MAX_LEN,
        add_special_tokens = True
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Determine where the answer starts in the (possibly truncated) sequence
    # If prompt got truncated, we cap at sequence length.
    ans_start = min(len(prompt_ids), len(input_ids))

    # Initialize labels as a copy of input_ids, then mask as needed
    labels = input_ids.copy()

    # Mask the prompt (everything before answer start)
    # This ensures loss is computed only on the answer part
    for i in range(ans_start):
        labels[i] = -100

    # Mask padding positions
    pad_id = tok.pad_token_id
    labels = [(-100 if tid == pad_id else tid) for tid in labels]

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
    }


# Tokenizing the dataset
tokenized = dataset.map(
    tokenize_with_labels,
    remove_columns = dataset.column_names
)

# Initializing the pre-trained model
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Resizing the token embeddings to match the tokenizer vocabulary size
model.resize_token_embeddings(len(tok))

# Moving the model to CPU
device = torch.device("cuda")\
    if (USE_CUDA and torch.cuda.is_available())\
    else torch.device("cpu")
model.to(device)

# Data collator for language modeling task.
# Data collator will dynamically pad the inputs received,
# as well as create the labels tensor.
# Padding example: [1, 2, 3] -> [1, 2, 3, 0, 0] if max length in batch is 5.
# We set mlm = False for causal language modeling.
collator = DataCollatorForLanguageModeling(tok, mlm = False)

# Training arguments for the Trainer
training_args = TrainingArguments(
    # Output directory for the fine-tuned model
    output_dir = TUNED_MODEL,
    # Number of training epochs
    num_train_epochs = 5,
    # Batch sizes for training and evaluation
    per_device_train_batch_size = 1,
    per_device_eval_batch_size = 1,
    gradient_accumulation_steps = 1,
    dataloader_num_workers = 1,
    # Learning rate and other hyperparameters
    learning_rate = 2e-5,
    weight_decay = 0.01,
    warmup_ratio = 0.03,
    logging_steps = 10,
    # Save strategy: no saving during training
    save_strategy = "no",
    # Disabling CUDA if not used
    use_cpu = (not USE_CUDA),
    fp16 = (USE_CUDA and torch.cuda.is_available()),
    bf16 = False,
    # Reporting settings
    report_to = ["tensorboard"],
    logging_dir = os.path.join(TUNED_MODEL, "logs"),
)

# Trainer initialization
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = tokenized,
    processing_class = tok,
    data_collator = collator,
)

# Instructions to run TensorBoard for monitoring
print("========================================")
print("Run tensorboard to monitor training:")
print(f"tensorboard --logdir {os.path.join(TUNED_MODEL, 'logs')} --bind_all")
print("========================================")

# Training the model
print("Starting training...")
trainer.train()
print("Training completed successfully!")

# Saving the fine-tuned model and tokenizer
trainer.save_model(TUNED_MODEL)

# The tokenizer is saved to ensure that the fine-tuned model can be used correctly in the future.
# The tokenizer is responsible for converting text into a format that the model can understand (i.e., token IDs)
# and vice versa.
# If the tokenizer is not saved along with the model,
# there may be inconsistencies in how text is processed during inference,
# especially if the tokenizer has been modified or if a different version of the tokenizer is used later.
# By saving both the model and the tokenizer together,
# we ensure that they remain compatible and
# that the model can be used effectively for generating predictions on new data.
tok.save_pretrained(TUNED_MODEL)
print(f"Model and tokenizer saved to {TUNED_MODEL}")
