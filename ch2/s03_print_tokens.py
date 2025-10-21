from transformers import AutoTokenizer

# List of models
models = [
    'facebook/bart-base',
    'Qwen/Qwen2.5-0.5B-Instruct'
]

text = "Why are ya doing this? ! =("

for model in models:
    tokenizer = AutoTokenizer.from_pretrained(model)
    # Tokenize the input
    tokens = tokenizer(text, return_tensors = "pt")["input_ids"][0]
    # Decode tokens one by one
    decoded_tokens = [tokenizer.decode([token]) for token in tokens]

    print(f"Model: {model}")
    print("Token IDs:", tokens.tolist())
    print("Decoded Tokens:", decoded_tokens)
    print('-----------')

# Model: facebook/bart-base
# Token IDs: [0, 7608, 32, 13531, 608, 42, 116, 27785, 5457, 1640, 2]
# Decoded Tokens: ['<s>', 'Why', ' are', ' ya', ' doing', ' this', '?', ' !', ' =', '(', '</s>']
# -----------
# Model: Qwen/Qwen2.5-0.5B-Instruct
# Token IDs: [10234, 525, 13526, 3730, 419, 30, 753, 43606]
# Decoded Tokens: ['Why', ' are', ' ya', ' doing', ' this', '?', ' !', ' =(']
# -----------
