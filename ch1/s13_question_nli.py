from transformers import pipeline

classifier = pipeline(
    "text-classification",
    # Model size: 418M
    model = "cross-encoder/qnli-electra-base"
)

question = "What is the capital of Germany?"
context = "Berlin is the largest city in Europe."

response = classifier(
    # Separating the premise and hypothesis with </s> token
    f"{question} [SEP] {context}",
    return_all_scores = True
)

# Map the label to human-readable format
label_mapping = {"LABEL_0": "not entailment", "LABEL_1": "entailment"}
response_named = [
    {"label": label_mapping[item["label"]], "score": item["score"]}
    for item in response[0]
]

print(response_named)
# [
# {'label': 'not entailment', 'score': 0.9529606699943542}
# ]
