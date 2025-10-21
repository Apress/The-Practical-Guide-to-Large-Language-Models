from transformers import AutoTokenizer

# Create a chat history
chat = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
    {"role": "user", "content": "Give me a positive affirmation for today!"},
    # This message will be continued
    {"role": "assistant", "content": "Ok! Here you go:"}
]

# Initialize the tokenizer for the model
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Apply the chat template with and without continue_final_message = False
print("Chat template with continue_final_message set to False:")
print(tokenizer.apply_chat_template(
    chat,
    tokenize = False,
    continue_final_message = False
))

# Apply the chat template with continue_final_message = True
print("Chat template with continue_final_message set to True:")
print(tokenizer.apply_chat_template(
    chat,
    tokenize = False,
    continue_final_message = True
))
