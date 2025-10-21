from transformers import AutoModel
model = AutoModel.from_pretrained("facebook/bart-large-mnli")

print(type(model))
