# ==== PDF Loader ====
# Path to the PDF file
# Defining a local file path or a URL
pdf_path = "https://arxiv.org/pdf/1706.03762"

# Load the PDF file using PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader

pdf_loader = PyPDFLoader(pdf_path)

# Load the pages from the PDF
pages = pdf_loader.load()

# Print the content of the first page
print(pages[0].page_content)

# ==== DOCX Loader ====
# Defining a local file path or a URL for the DOCX file
docx_path = "https://calibre-ebook.com/downloads/demos/demo.docx"

# Load the DOCX file using UnstructuredWordDocumentLoader
from langchain_community.document_loaders import Docx2txtLoader

loader = Docx2txtLoader(docx_path)

# Loading the DOCX file
docs = loader.load()

# Print the content of the first document
print(docs[0].page_content)

# ==== Wikipedia Loader ====
# Load a Wikipedia page using the WikipediaLoader
from langchain_community.document_loaders import WikipediaLoader

wiki_loader = WikipediaLoader(
    query = "Disney Animation Studios",
    lang = "en",
    load_max_docs = 10
)

# Load the pages from Wikipedia
wiki_docs = wiki_loader.load()

# Print the content of the first Wikipedia document
print(wiki_docs[0].page_content)

# ==== Web Loader ====
from langchain_community.document_loaders import WebBaseLoader

# Load a web page using the WebBaseLoader
web_loader = WebBaseLoader("https://en.wikipedia.org/wiki/The_Lion_King")

# Load the pages from the web
web_docs = web_loader.load()

# Print the content of the first web document
print(web_docs[0].page_content)

# ==== Arxiv Loader ====
from langchain_community.document_loaders import ArxivLoader

# Initialize the ArxivLoader with a search query
arxiv_loader = ArxivLoader(
    query = "machine learning",
    doc_content_chars_max = 1000
)

docs = loader.load()
print(docs[0].page_content[:100])
print(docs[0].metadata)

# ==== Hugging Face Model Loader ====
from langchain_community.document_loaders import HuggingFaceModelLoader

# Initialize the HuggingFaceModelLoader with search criteria
hf_loader = HuggingFaceModelLoader(
    search = "bert",
    limit = 5
)

# Load models from Hugging Face
hf_docs = hf_loader.load()

# Print the content of the first Hugging Face model document
print(hf_docs[0].page_content)
