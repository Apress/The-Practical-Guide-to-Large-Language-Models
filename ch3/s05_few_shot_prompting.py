from transformers import AutoTokenizer, AutoModelForCausalLM

# Defining preprompt
chat = [
    {"role": "user", "content": "What is the most popular movie starring Marilyn Monroe?"},
    {"role": "assistant", "content": "Some Like It Hot (1959)"}
]

# Adding original prompt
chat.append(
    {"role": "user", "content": "What was the first movie Charlie Chaplin appeared in?"}
)

# Initializing the model and tokenizer
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Prepare prompt
prompt = tokenizer.apply_chat_template(
    chat,
    tokenize = False,
    add_generation_prompt = True
)

# Tokenize input
inputs = tokenizer(prompt, return_tensors = "pt")

# Generate output
outputs = model.generate(
    **inputs,
    max_new_tokens = 100,
    do_sample = True,
    temperature = 0.7
)

# Decode and print response
response = tokenizer.decode(
    outputs[0][inputs['input_ids'].shape[-1]:],
    skip_special_tokens = True
)

# Print the response
print(response)
# The Kid (1921)
