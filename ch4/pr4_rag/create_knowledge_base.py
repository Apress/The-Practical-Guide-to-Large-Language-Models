# Get most downloadable models from Hugging Face
import os
from time import sleep
from huggingface_hub import list_models
from langchain_community.document_loaders import HuggingFaceModelLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# List the most downloaded models for text generation
print("Getting the most downloaded models for text generation...")
total_models = 500
models = list(list_models(
    filter = "text-generation",
    sort = "downloads",
    direction = -1,
    limit = total_models
))

# Creating a set to store unique model IDs
unique_model_ids = set()
# Iterate through the models and add their IDs to the set
for model in models:
    unique_model_ids.add(model.id)

print("Downloading model documents from Hugging Face...")
docs = {}
counter = 0
for model_id in unique_model_ids:
    counter += 1
    # Create a HuggingFaceModelLoader for each model ID
    loader = HuggingFaceModelLoader(
        search = model_id,
        limit = 1
    )
    # Load the model document
    doc = loader.load()[0]
    # Loading only unique documents
    docs[doc.metadata['_id']] = doc

    if counter % 50 == 0:
        print(f"Loaded {counter} models...")
        # Sleep for a short time to avoid hitting API limits
        sleep(3)

print("Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 100,
    length_function = len,
    is_separator_regex = False,
    separators = ["\n\n", "\n", " ", ""],

)
chunks = text_splitter.split_documents(docs.values())

print("Creating embeddings for the chunks and storing them in FAISS...")
embedding_model = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

current_file_path = os.path.dirname(os.path.abspath(__file__))
storage_dir = f"{current_file_path}/hf_models_index"

vectorstore = FAISS.from_documents(chunks, embedding_model)
vectorstore.save_local(storage_dir)
