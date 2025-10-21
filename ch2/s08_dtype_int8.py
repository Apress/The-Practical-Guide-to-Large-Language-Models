from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


def vram_usage_gb():
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**3)
    return 0.0


model_id = "Qwen/Qwen2.5-0.5B-Instruct"

# Define 8-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_8bit = True,
    llm_int8_threshold = 6.0,
    llm_int8_skip_modules = None,
    llm_int8_enable_fp32_cpu_offload = False
)

print(f"VRAM before loading: {vram_usage_gb():.2f} GB")

# Load the model with 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config = bnb_config,
    device_map = "cuda"
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Print VRAM usage after loading the model with 8-bit quantization
print(f"VRAM after loading: {vram_usage_gb():.2f} GB")
# VRAM after loading: 0.59 GB
