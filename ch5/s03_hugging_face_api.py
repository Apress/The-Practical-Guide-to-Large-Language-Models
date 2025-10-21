from huggingface_hub import InferenceClient

# https://huggingface.co/settings/tokens
hf_token = 'your_token'

client = InferenceClient(
    api_key = hf_token,
)

completion = client.chat.completions.create(
    model = "meta-llama/Llama-3.1-8B-Instruct",
    messages = [
        {
            "role":    "user",
            "content": "How many parameters does the Llama-3.1-8B model have?"
        }
    ],
)

print(completion.choices[0].message)
