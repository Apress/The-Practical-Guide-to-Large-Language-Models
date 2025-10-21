# Importing necessary libraries
import json
import os
from pathlib import Path
from datasets import load_dataset
from langchain_community.document_loaders import WikipediaLoader


# Function to load Wikipedia pages and save them to a JSONL file
def wikipedia_to_jsonl(
        topics,
        out_path: str,
        lang: str = "en",
        load_max_docs: int = 10,
):
    # Initialize the WikipediaLoader with the specified parameters
    loader = WikipediaLoader(
        query = topics,
        lang = lang,
        load_max_docs = load_max_docs,
    )

    # Load the documents
    docs = loader.load()

    # Creating the output directory if it does not exist
    out_path = Path(out_path)
    out_path.parent.mkdir(parents = True, exist_ok = True)

    # Writing the documents to a JSONL file
    n_written = 0
    with out_path.open("w", encoding = "utf-8") as f:
        for i, d in enumerate(docs):
            record = {
                "id":     str(i),
                "title":  d.metadata.get("title") or "untitled",
                "source": d.metadata.get("source") or "",
                "text":   d.page_content.strip() if d.page_content else "",
            }
            f.write(json.dumps(record, ensure_ascii = False) + "\n")
            n_written += 1

    print(f"Wrote {n_written} records to {out_path}")


# Creating a JSONL file with Wikipedia pages about Large Language Models
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(SCRIPT_DIR, "data", "wikipedia_llm.jsonl")
wikipedia_to_jsonl(
    topics = "Large Language Models",
    out_path = dataset_path,
    lang = "en",
    load_max_docs = 20
)

# loading the dataset from the JSONL file using the datasets library
llm_ds = load_dataset("json", data_files = dataset_path)
