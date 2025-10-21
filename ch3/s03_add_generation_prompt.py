from transformers import AutoTokenizer

# Define a chat history
chat = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
    {"role": "user", "content": "Give me a positive affirmation for today!"},
]

# Initialize the tokenizer for the model
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Apply the chat template with and without add_generation_prompt = False
print("Chat template with add_generation_prompt set to False:")
print(tokenizer.apply_chat_template(
    chat,
    tokenize = False,
    add_generation_prompt = False
))

# Apply the chat template with add_generation_prompt set to True
print("Chat template with add_generation_prompt set to True:")
print(tokenizer.apply_chat_template(
    chat,
    tokenize = False,
    add_generation_prompt = True
))
