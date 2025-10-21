import torch
from transformers import AutoTokenizer

# Loading the tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# Creating a tensor with token IDs
token_ids = torch.tensor(
    [[4340, 525, 498, 30]],
    dtype = torch.long
)

# Decoding the token IDs to text
decoded_text = tokenizer.decode(
    token_ids[0].tolist(),
    skip_special_tokens = True
)

print(decoded_text)
# How are you?
