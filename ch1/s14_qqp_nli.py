from transformers import pipeline

classifier = pipeline(
    "text-classification",
    # Model size: 1639M
    model = "howey/roberta-large-qqp"
)

q1 = "What is the best way to learn Python?"
q2 = "How can I learn Python programming?"

response = classifier(
    f"{q1} [SEP] {q2}",
    return_all_scores = True
)

# Map the label to human-readable format
label_mapping = {
    "LABEL_0": "not entailment",
    "LABEL_1": "entailment"
}
response_named = [
    {
        "label": label_mapping[item["label"]],
        "score": item["score"]
    }
    for item in response[0]
]

print(response_named)

# [
# {'label': 'not entailment', 'score': 0.20287956297397614},
# {'label': 'entailment', 'score': 0.7971204519271851}
# ]
