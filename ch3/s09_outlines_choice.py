from typing import Literal
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer

# Defining engine LLM model
model_id = "Qwen/Qwen2.5-0.5B-Instruct"

# Loading model wrapped by Outlines
model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(model_id),
    AutoTokenizer.from_pretrained(model_id)
)

# Define the prompt
prompt = "Pick the odd word out from: dog, cat, mouse, sun\nAnswer:"

# Creating the outlines model for structured generation
generator = outlines.Generator(
    model,
    Literal["dog", "cat", "mouse", "sun"]
)

# Generate structured answer
structured_answer = generator(prompt)

print("Structured answer:")
print(structured_answer)
# sun
