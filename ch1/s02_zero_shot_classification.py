from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    # Model size: 1557M
    model = "facebook/bart-large-mnli"
)

text = "German football team wins the World Cup!"
labels = ["sports", "politics", "business"]

result = classifier(text, candidate_labels = labels)

print(result)

# {
# 'sequence': 'German football team wins the World Cup!',
# 'labels': ['sports', 'business', 'politics'],
# 'scores': [0.9958564043045044, 0.003022331977263093, 0.001121298992075026]
# }
