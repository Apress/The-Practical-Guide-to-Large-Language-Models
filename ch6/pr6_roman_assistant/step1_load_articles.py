# pip install wikipedia-api==0.8.1

# Importing necessary libraries
import os
import re
from pathlib import Path
import wikipediaapi

# We will download articles in English about the Roman Empire
LANG = "en"
TOPICS = [
    "Roman Empire",
    "Ancient Rome",
    "Roman Republic",
    "Julius Caesar",
    "Augustus",
    "Pax Romana",
    "Roman law",
    "Roman army",
    "Roman legion",
    "Roman architecture",
    "Roman engineering",
    "Roman roads",
    "Roman province",
    "Byzantine Empire",
    "Fall of the Western Roman Empire",
    "Diocletian",
    "Constantine the Great",
    "Senate of the Roman Empire",
    "Latin",
    "Roman religion",
]

# Output directory for the downloaded articles
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "data", "roman_empire_txt")

# Minimum number of characters to save the article.
# Too short articles are skipped.
MIN_CHARS = 500
# If True, add a header with the article title and source URL
ADD_HEADER = True

# Wikipedia API client
wiki = wikipediaapi.Wikipedia(
    language = LANG,
    extract_format = wikipediaapi.ExtractFormat.WIKI,
    user_agent = "wiki-downloader/1.0 (contact@example.com)",
    timeout = 15,
)


# Function to create a filesystem-friendly slug from the article title
def _slugify(name: str, max_len: int = 100) -> str:
    s = (name or "untitled").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    if len(s) > max_len:
        s = s[:max_len].rstrip("-")
    return s or "untitled"


# Function to save a Wikipedia page to a text file
def save_page(title: str, out_dir: Path) -> bool:
    page = wiki.page(title)
    if not page.exists():
        print(f"[skip] Page not found: {title!r}")
        return False

    text = (page.text or "").strip()
    if len(text) < MIN_CHARS:
        print(f"[skip] Too short ({len(text)} chars): {title!r}")
        return False

    fname = f"{_slugify(page.title)}.txt"
    path = out_dir / fname

    k = 1
    while path.exists():
        path = out_dir / f"{_slugify(page.title)}_{k}.txt"
        k += 1

    with path.open("w", encoding = "utf-8") as f:
        if ADD_HEADER:
            f.write(f"# {page.title}\n")
            if hasattr(page, "fullurl") and page.fullurl:
                f.write(f"Source: {page.fullurl}\n\n")
        f.write(text)

    print(f"[ok] Saved: {page.title} -> {path}")
    return True


# Create output directory if it doesn't exist
out_path = Path(OUT_DIR)
out_path.mkdir(parents = True, exist_ok = True)

# Main loop to download and save articles
n = 0
for t in TOPICS:
    try:
        if save_page(t, out_path):
            n += 1
    except Exception as e:
        print(f"[error] {t!r}: {e}")

# Final report
print(f"\nDone. Saved {n} articles to {out_path}")
