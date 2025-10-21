from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
from transformers import TextIteratorStreamer
import threading
import torch

# Constants
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CHAT_HISTORY = 10
MAX_NEW_TOKENS = 1000

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype = torch.float16,
    device_map = "auto"
)


# Generate response function
def generate_response(message, history):
    # Creating a list of messages for the chat
    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    for msg in history[-MAX_CHAT_HISTORY:]:
        messages.append({"role": msg['role'], "content": msg['content']})

    messages.append({"role": "user", "content": message})

    # Convert messages to text
    prompt = tokenizer.apply_chat_template(
        messages, tokenize = False, add_generation_prompt = True
    )

    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors = "pt").to(device)

    # Create a text streamer
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt = True,
        skip_special_tokens = True
    )

    # Generate response in a separate thread
    generation_kwargs = dict(
        **inputs,
        streamer = streamer,
        max_new_tokens = MAX_NEW_TOKENS,
        pad_token_id = tokenizer.eos_token_id
    )

    thread = threading.Thread(target = model.generate, kwargs = generation_kwargs)
    thread.start()

    output = []
    for token in streamer:
        output.append(token)
        yield ''.join(output)


# Gradio interface
demo = gr.ChatInterface(
    fn = generate_response,
    type = "messages",
    title = "Your Personal Chat Assistant",
    description = "Interactive chat. Ask questions, and I will try to answer them!",
    examples = [
        "How to cook a perfect steak?",
        "What are the benefits of meditation?",
        "Can you recommend a good book on machine learning?",
    ]
)

# Launching the Gradio app
demo.launch(share = True)
