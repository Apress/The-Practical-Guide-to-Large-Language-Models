import os
import torch
from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define constants
# Model size: 1GB
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_CHAT_HISTORY = 5
MAX_NEW_TOKENS = 300

# Load the model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype = "auto",
    device_map = "auto"
).to(device)

# Initialize chat history
chat_history = []


# Generate response based on user prompt
def generate_response(prompt):
    global chat_history

    # Add a system message to define the assistant
    messages = [
        {
            "role":    "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        }
    ]

    # Add the chat history to the context
    for entry in chat_history[-MAX_CHAT_HISTORY:]:  # Include only the last 5 entries in the chat history
        role, content = entry.split(": ", 1)
        messages.append({
            "role":    role,
            "content": content
        })

    # Add the last user prompt
    messages.append({"role": "user", "content": prompt})

    # unify the chat messages into a standardized text format for the model
    text = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)

    # tokenize the text
    model_inputs = tokenizer([text], return_tensors = "pt").to(device)

    # generate the response
    outputs = model.generate(
        **model_inputs,
        max_new_tokens = MAX_NEW_TOKENS,
        pad_token_id = tokenizer.eos_token_id
    )
    generated_ids = outputs[:, model_inputs.input_ids.shape[-1]:]

    # Decode the response
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens = True)[0]

    # Append request and response to chat history
    chat_history.append(f"user: {prompt}")
    chat_history.append(f"assistant: {response}")

    return response


# Initialize Flask app
# Define the path to the templates directory
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
if not os.path.exists(TEMPLATE_DIR):
    raise FileNotFoundError(f"Templates directory not found at {TEMPLATE_DIR}")
app = Flask(__name__, template_folder = TEMPLATE_DIR)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods = ["POST"])
def chat():
    user_message = request.json["message"]
    bot_response = generate_response(user_message)
    return jsonify({"response": bot_response})


if __name__ == "__main__":
    app.run(debug = True)
