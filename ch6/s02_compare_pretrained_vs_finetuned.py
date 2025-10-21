# Importing necessary libraries
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Original pre-trained model and output directory for fine-tuned model
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TUNED_MODEL_DIR = "/tmp/qwen-alien_romulus-finetuned"

# Path to the dataset file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "alien_romulus_qa.jsonl")

# Loading the dataset from JSONL file
dataset = load_dataset(
    "json",
    data_files = DATA_PATH,
    split = "train"
)

# Taking a list of unique instructions
PROMPTS = dataset.unique("instruction")[:20]


# Function to generate text using the model
def gen(model_name, prompts, max_new_tokens = 80):

    # Loading the tokenizer and model
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    # Iterate over prompts and generate responses
    for p in prompts:
        inp = tok(f"Question: {p}\nAnswer:", return_tensors = "pt")
        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens = max_new_tokens,
                pad_token_id = tok.eos_token_id,
            )
        print("=" * 80)
        print(tok.decode(out[0], skip_special_tokens = True))


# Generating responses before and after fine-tuning
print("\n--- BEFORE (pre-trained) ---")
gen(MODEL_NAME, PROMPTS)

print("\n--- AFTER (fine-tuned) ---")
gen(TUNED_MODEL_DIR, PROMPTS)
