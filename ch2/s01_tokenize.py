from transformers import AutoTokenizer

# Loading the tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# Tokenizing the text
sentence = "How are you?"
tokens = tokenizer(sentence, return_tensors = "pt")

# Result
print("Token id list:", tokens["input_ids"])
