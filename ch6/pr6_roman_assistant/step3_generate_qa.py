# pip install huggingface_hub

# Importing necessary libraries
import os
import re
import json
import random
import time
from pathlib import Path
from typing import Optional, Tuple
from huggingface_hub import InferenceClient


# Inference client wrapper to call the model on Hugging Face Hub
class InferenceClientModel:

    def __init__(self, model_id: str, api_key: str):
        self.client = InferenceClient(model = model_id, token = api_key)

    def chat(
            self,
            messages,
            max_tokens: int = 256,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            stop: Optional[list] = None,
    ) -> str:
        # Single chat completion call (non-streaming)
        resp = self.client.chat_completion(
            messages = messages,
            max_tokens = max_tokens,
            temperature = temperature,
            top_p = top_p,
            stop = stop,
            stream = False,
        )
        # Hugging Face returns a dict with 'choices' key
        try:
            return resp.choices[0].message["content"]
        except Exception:
            # If something is wrong, return the full response for debugging
            return str(resp)


# Remote model parameters.
# This is conversational model, so we use chat completions.
MODEL_ID = "NousResearch/Hermes-3-Llama-3.1-405B"
# Your Hugging Face token from https://huggingface.co/settings/tokens
hf_token = os.getenv("HF_TOKEN", "your_token")

# Model client
hf_model = InferenceClientModel(model_id = MODEL_ID, api_key = hf_token)

# Paths and parameters
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTICLES_DIR = Path(SCRIPT_DIR) / "data" / "roman_empire_txt"
OUT_PATH = Path(SCRIPT_DIR) / "data" / "roman_qa.jsonl"

NUM_EXAMPLES = 150
PARA_MIN_CHARS = 300
PARA_MAX_CHARS = 1200
SLEEP_BETWEEN_CALLS = (0.25, 0.7)
SHUFFLE_SEED = 42

# Requirements for generated Q/A:
# Min tokens in question
MIN_Q_TOKENS = 7
# Min/max chars in answer
MIN_ANS_CHARS = 60
MAX_ANS_CHARS = 450
# Forbidden patterns in question.
# We want non-trivial questions, not "What is the name of ...?".
FORBIDDEN_QUESTION_PATTERNS = [
    r"\bwhat is the name of\b",
    r"\bwhat is the name\b",
    r"\bwhat\s+is\s+the\s+language\b",
    r"\bwhat\s+is\s+the\s+dynasty\b",
]


# Functions for reading paragraphs from text files.
# We split by empty lines and filter by length.
def read_paragraphs_from_dir(indir: Path):
    files = sorted(indir.glob("*.txt"))
    for fp in files:
        try:
            text = fp.read_text(encoding = "utf-8")
        except UnicodeDecodeError:
            text = fp.read_text(encoding = "utf-8", errors = "ignore")
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        for i, p in enumerate(paras):
            if len(p) >= PARA_MIN_CHARS:
                yield {"file": str(fp), "para_id": i, "text": p}


# Function to pick N random paragraphs from the directory
def pick_paragraphs(indir: Path, n: int):
    pool = list(read_paragraphs_from_dir(indir))
    if not pool:
        raise RuntimeError(f"No paragraphs >= {PARA_MIN_CHARS} chars in {indir}")
    random.Random(SHUFFLE_SEED).shuffle(pool)
    return pool[:n] if n < len(pool) else pool


# Truncate paragraph to max length, preferably at sentence ending.
def truncate_paragraph(p: str) -> str:
    if len(p) <= PARA_MAX_CHARS:
        return p
    cut = p[:PARA_MAX_CHARS]
    last_dot = cut.rfind(".")
    if last_dot > int(PARA_MAX_CHARS * 0.6):
        return cut[: last_dot + 1].strip()
    return cut.strip()


# Building messages for the chat model
def build_messages(paragraph: str):
    # System prompt: strict format in JSON
    system = (
        'Output ONLY one minified JSON object: {"question":"...","answer":"..."} '
        "No code fences. No extra text."
    )
    user = (
        "Write ONE question about the paragraph and answer it using ONLY the paragraph. "
        "The answer MUST be a verbatim substring (continuous span) of the paragraph.\n\n"
        "Paragraph:\n<<<\n"
        f"{paragraph}\n"
        ">>>"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# Parsing and validating the model output
def parse_qa(s: str) -> Optional[Tuple[str, str]]:
    # Extract JSON object from the text
    s = s.lstrip(" \n\r\t'\"")
    m = re.search(r"\{.*\}", s, flags = re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    q = (obj.get("question") or "").strip()
    a = (obj.get("answer") or "").strip()
    return (q, a) if q and a else None


# Validation functions
def token_count(text: str) -> int:
    return len([t for t in re.findall(r"\w+", text)])


def has_forbidden_pattern(q: str) -> bool:
    q_low = q.lower()
    return any(re.search(p, q_low) for p in FORBIDDEN_QUESTION_PATTERNS)


def has_verb(text: str) -> bool:
    # Simple heuristic: look for common English verbs or -ed/-ing endings
    return bool(
        re.search(
            r"\b(is|are|was|were|be|been|being|has|have|had|do|does|did|can|could|may|might|must|should|will|would|["
            r"a-z]{3,}ed|[a-z]{3,}ing)\b",
            text,
            flags = re.I,
        )
    )


# A sentence-like answer ends with ., !, or ? and has at least one space
def is_sentence_like(ans: str) -> bool:
    return ans.endswith((".", "!", "?")) and (" " in ans)


# Full validation of Q/A pair
def validate_qa(q: str, a: str, para: str) -> bool:
    if a not in para:
        return False
    if len(a) < MIN_ANS_CHARS or len(a) > MAX_ANS_CHARS:
        return False
    if token_count(q) < MIN_Q_TOKENS:
        return False
    if has_forbidden_pattern(q):
        return False
    if not has_verb(a) or not is_sentence_like(a):
        return False
    return True


# Generate one Q/A pair with retries
def gen_one_qa(paragraph: str, attempts: int = 3) -> Optional[Tuple[str, str]]:
    msgs = build_messages(paragraph)
    for att in range(attempts):
        out = hf_model.chat(
            messages = msgs,
            max_tokens = 220,
            # Use lower temperature on retries
            temperature = 0.2 if att == 0 else 0.5,
            top_p = 0.9,
            # In case model goes rogue
            stop = ["```", "</s>"],
        )
        qa = parse_qa(out)
        if not qa:
            continue
        q, a = qa
        if validate_qa(q, a, paragraph):
            return q, a
    return None


# Paragraphs to process
paras = pick_paragraphs(ARTICLES_DIR, NUM_EXAMPLES * 4)

n_written = 0
seen_q = set()

# Main loop: generate and write Q/A pairs
with OUT_PATH.open("w", encoding = "utf-8") as f:
    for item in paras:
        para = truncate_paragraph(item["text"])
        try:
            qa = gen_one_qa(para, attempts = 3)
        except Exception as e:
            print(f"[error] gen failed for {item['file']}#{item['para_id']}: {e}")
            qa = None

        if not qa:
            time.sleep(random.uniform(*SLEEP_BETWEEN_CALLS))
            continue

        q, a = qa
        if q in seen_q:
            continue
        seen_q.add(q)

        rec = {
            "instruction": q,
            "response":    a,
            "meta":        {
                "source_file":   os.path.relpath(item["file"], SCRIPT_DIR),
                "paragraph_id":  item["para_id"],
                "paragraph_len": len(para),
                "answer_span":   [para.find(a), para.find(a) + len(a)],
            },
        }
        f.write(json.dumps(rec, ensure_ascii = False) + "\n")
        n_written += 1
        print(f"[ok] {n_written:04d}: {q[:80]}")

        if n_written >= NUM_EXAMPLES:
            break

        time.sleep(random.uniform(*SLEEP_BETWEEN_CALLS))

print(f"\nDone. Wrote {n_written} records to {OUT_PATH}")
