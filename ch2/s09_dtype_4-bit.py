from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


def vram_usage_gb():
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**3)
    return 0.0


model_id = "Qwen/Qwen2.5-0.5B-Instruct"

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_quant_type = "nf4",  # "fp4" is also possible
    bnb_4bit_compute_dtype = torch.float16
)

print(f"VRAM before loading: {vram_usage_gb():.2f} GB")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config = bnb_config,
    device_map = "cuda"
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

print(f"VRAM after loading: {vram_usage_gb():.2f} GB")
# VRAM after loading: 0.43 GB
