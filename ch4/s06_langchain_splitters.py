import requests
from langchain_community.document_loaders import PyPDFLoader

pdf_path = "https://arxiv.org/pdf/1706.03762"
pdf_loader = PyPDFLoader(pdf_path)
pages = pdf_loader.load()
text = '\n'.join([page.page_content for page in pages])

# ==== CharacterTextSplitter ====
from langchain_text_splitters import CharacterTextSplitter

char_text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 2000,
    chunk_overlap = 200,
    length_function = len,
)

char_texts = char_text_splitter.split_text(text)

for i, t in enumerate(char_texts):
    print(f"Chunk {i + 1}:")
    print(t)
    print('====================')

# ==== Recursive Text Splitter ====
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 2000,
    chunk_overlap = 200,
    length_function = len,
    is_separator_regex = False,
    separators = ["\n\n", "\n", " ", ""],
)
chunks = text_splitter.split_text(text)

for i, t in enumerate(chunks):
    print(f"Chunk {i + 1}:")
    print(t)
    print('====================')

# ==== HTMLHeaderTextSplitter ====

html = """
<h1>Machine Learning</h1>
<p>Introduction to ML...</p>
<h2>Linear Regression</h2>
<p>Model description...</p>
<h1>Deep Learning</h1>
<p>Neural networks, etc.</p>
"""

from langchain_text_splitters import HTMLHeaderTextSplitter

html_splitter = HTMLHeaderTextSplitter(
    headers_to_split_on = [
        ("h1", "Main Topic"),
        ("h2", "Sub Topic")
    ],
    return_each_element = False
)

html_texts = html_splitter.split_text(html)

# ==== SentenceTransformersTokenTextSplitter ====
from langchain_text_splitters import SentenceTransformersTokenTextSplitter

token_text_splitter = SentenceTransformersTokenTextSplitter(
    model_name = "all-MiniLM-L6-v2",
    chunk_size = 200,
    chunk_overlap = 20,
    length_function = len,
)

token_texts = token_text_splitter.split_text(text)
