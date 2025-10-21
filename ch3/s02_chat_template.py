from transformers import AutoTokenizer

# Define a chat history
chat = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
    {"role": "user", "content": "Give me a positive affirmation for today!"},
]

# Define two different models to compare their chat templates
MODEL_NAME_1 = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_NAME_2 = "microsoft/Phi-3-mini-4k-instruct"

# Load the tokenizers for both models
tokenizer1 = AutoTokenizer.from_pretrained(MODEL_NAME_1)
tokenizer2 = AutoTokenizer.from_pretrained(MODEL_NAME_2)

# Apply the chat template to Qwen model
print("Chat template for Qwen:")
print(tokenizer1.apply_chat_template(chat, tokenize = False))

# Apply the chat template to Phi-3 model
print("\nChat template for Phi-3:")
print(tokenizer2.apply_chat_template(chat, tokenize = False))
