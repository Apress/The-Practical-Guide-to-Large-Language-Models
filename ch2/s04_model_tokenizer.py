import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load the tokenizer and model
model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Tokenize the input text
text = "I like Large Language Models very much!"
inputs = tokenizer(text, return_tensors = "pt")

# Run inference without gradient calculation
with torch.no_grad():
    outputs = model(**inputs)

# Apply softmax to get probability distribution
probs = torch.nn.functional.softmax(
    outputs.logits,
    dim = -1
)

# Map predicted class index to label
label_map = {0: "NEGATIVE", 1: "POSITIVE"}
predicted_label = label_map[torch.argmax(probs).item()]
confidence_score = probs.max().item()

# Print the result
response = [{"label": predicted_label, "score": confidence_score}]
print(response)

# [{'label': 'POSITIVE', 'score': 0.9997369647026062}]
