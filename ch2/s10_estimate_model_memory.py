def estimate_model_memory(num_params: int, dtype: str = "float16") -> float:
    dtype_bytes = {
        "float32":  4,
        "float16":  2,
        "bfloat16": 2,
        "int8":     1,
        "4bit":     0.5
    }

    if dtype not in dtype_bytes:
        raise ValueError(
            f"Unsupported dtype '{dtype}'. "
            f"Supported: {list(dtype_bytes.keys())}"
        )

    total_bytes = num_params * dtype_bytes[dtype]
    total_gb = total_bytes / (1024**3)
    return total_gb


params = 7_000_000_000  # 7 billion parameters
dtype = "int8"
size_gb = estimate_model_memory(params, dtype)

print(f"{params / 1e6:.1f}M params in {dtype}: {size_gb:.2f} GB")
# 7000.0M params in int8: 6.52 GB
