import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Example text chunks to create embeddings
chunks = [
    "Short text example to demonstrate the embedding model.",
    "Another example text that will be used to create embeddings.",
]

# Initialize the HuggingFace embedding model
embedding_model = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

# Create embeddings for the chunks
embeddings = embedding_model.embed_documents(chunks)

storage_dir = "foo_index"

# Ensure the storage directory exists
if not os.path.exists(storage_dir):
    # Creating the storage for the first time
    vectorstore = FAISS.from_texts(chunks, embedding_model)
    vectorstore.save_local(storage_dir)
else:
    # Loading the existing vectorstore
    vectorstore = FAISS.load_local(
        storage_dir,
        embedding_model,
        # This is needed to allow loading of the vectorstore
        # This is safe if you created the vectorstore yourself
        allow_dangerous_deserialization = True
    )

# Perform a similarity search
context_docs = vectorstore.similarity_search(
    "some message",
    k = 1
)
