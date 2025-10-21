from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def vram_usage_gb():
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**3)
    return 0.0


model_id = "Qwen/Qwen2.5-0.5B-Instruct"

print(f"VRAM before loading: {vram_usage_gb():.2f} GB")
# VRAM before loading: 0.00 GB

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype = torch.float32,
    device_map = "auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

print(f"VRAM after loading: {vram_usage_gb():.2f} GB")
# VRAM after loading: 1.84 GB
