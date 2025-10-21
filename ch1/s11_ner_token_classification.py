from transformers import pipeline

classifier = pipeline(
    "ner",
    # Model size: 414M
    model = "dslim/bert-base-NER"
)

text = "Jack sent me 100 dollars from London by Western Union."

response = classifier(text)
print(response)

# [
# {'entity': 'B-PER', 'score': 0.99835986, 'index': 1, 'word': 'Jack', 'start': 0, 'end': 4},
# {'entity': 'B-LOC', 'score': 0.99959844, 'index': 7, 'word': 'London', 'start': 30, 'end': 36},
# {'entity': 'B-ORG', 'score': 0.9983321, 'index': 9, 'word': 'Western', 'start': 40, 'end': 47},
# {'entity': 'I-ORG', 'score': 0.9988926, 'index': 10, 'word': 'Union', 'start': 48, 'end': 53}
# ]
