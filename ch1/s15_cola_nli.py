from transformers import pipeline

classifier = pipeline(
    "text-classification",
    # Model size: 512M
    model = "textattack/distilbert-base-uncased-CoLA"
)

# Input sentence
text = "Running is enjoys she."

# Get model prediction
response = classifier(text)

# Map model labels to human-readable labels
label_mapping = {
    "LABEL_0": "Unacceptable (Grammatically Incorrect)",
    "LABEL_1": "Acceptable (Grammatically Correct)"
}

# Format output in a human-readable way
response_named = {
    "label": label_mapping[response[0]["label"]],
    "score": response[0]["score"]
}

print(response_named)

# {'label': 'Unacceptable (Grammatically Incorrect)', 'score': 0.9755550622940063}
