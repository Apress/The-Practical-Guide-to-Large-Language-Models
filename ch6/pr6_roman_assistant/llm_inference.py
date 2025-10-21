import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Tuned model directory
TUNED_MODEL_DIR = "/tmp/roman_assistant_model"

PROMPTS = [
    "What is the Roman Empire?",
    "How did the Roman Republic transition into the Roman Empire?",
    "Who was Augustus and why was his reign important?",
    "What role did the Roman Senate play under the Empire?",
    "How was the Roman army organized during the Imperial period?",
    "What were Roman provinces and how were they governed?",
    "How did Roman law influence later legal systems?",
    "What architectural features are characteristic of Roman buildings?",
    "What factors contributed to the fall of the Western Roman Empire?",
    "In what ways does Roman culture still influence the modern world?",
]


def gen(model_name_or_path, prompts):
    tok = AutoTokenizer.from_pretrained(model_name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    # Enable the cache for faster generation
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for i, q in enumerate(prompts, 1):
        prompt = f"Question: {q}\nAnswer:"
        inputs = tok(prompt, return_tensors = "pt").to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens = 200,
                eos_token_id = tok.eos_token_id,
                pad_token_id = tok.pad_token_id
            )

        print("=" * 80)
        print(f"[{i}] {tok.decode(out[0], skip_special_tokens = True)}")


gen(TUNED_MODEL_DIR, PROMPTS)


# Question: What role did the Roman Senate play under the Empire?
# Answer: The Senate held actual authority (auctoritas), but no real legislative power; it was technically only an
# advisory council.

# Question: What were Roman provinces and how were they governed?
# Answer: Roman provinces were governed by either the central government or provincial governors, who were appointed
# by the Senate.

# Question: How did Roman law influence later legal systems?
# Answer: The jurists of the post-classical period of the Roman Empire developed a large number of juridical texts,
# including the Justinian and Theodosian law codes.
