import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import gradio as gr
from transformers import TextIteratorStreamer
import threading
import torch

# Constants
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CHAT_HISTORY = 3
MAX_NEW_TOKENS = 1000

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_quant_type = "nf4",  # "fp4" is also possible
    bnb_4bit_compute_dtype = torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config = bnb_config,
    device_map = device,
    do_sample = True,
    temperature = 0.7,
)

current_file_path = os.path.dirname(os.path.abspath(__file__))
storage_dir = f"{current_file_path}/hf_models_index"
embedding_model = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    storage_dir,
    embedding_model,
    # This is needed to allow loading of the vectorstore
    # This is safe if you created the vectorstore yourself
    allow_dangerous_deserialization = True
)


# Generate response function
def generate_response(message, history):
    # Creating a list of messages for the chat
    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    for msg in history[-MAX_CHAT_HISTORY:]:
        messages.append({"role": msg['role'], "content": msg['content']})

    context_docs = vectorstore.similarity_search(
        message,
        k = 3
    )

    # Add context chunks to the messages
    context = "\n\n".join(doc.page_content for doc in context_docs)
    prompt =\
        f"Use the context below to answer the question. \n"\
        f"Context: {context} \n"\
        f"Question: {message} \n"\
        f"Assistant:"

    messages.append({"role": "user", "content": prompt})

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
    title = "Your Personal HuggingFace Assistant",
    description = "Ask questions about HuggingFace models",
    examples = [
        "What is DeepSeek?",
        "What is the context length of Phi-3-mini-128k-instruct model?",
        "How many parameters does Llama-3-8B model have?",
    ]
)

# Launching the Gradio app
demo.launch(share = True)
