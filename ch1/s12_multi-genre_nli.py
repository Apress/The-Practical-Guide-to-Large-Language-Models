from transformers import pipeline

classifier = pipeline(
    "text-classification",
    # Model size: 1363M
    model = "FacebookAI/roberta-large-mnli"
)

premise = "A scientist is working in a laboratory, analyzing data from a microscope"
hypothesis = "A researcher is conducting experiments in a lab."

response = classifier(
    # Separating the premise and hypothesis with </s> token
    f"{premise} </s> {hypothesis}",
    return_all_scores = True
)

print(response)

# [[
# {'label': 'CONTRADICTION', 'score': 0.0017340655904263258},
# {'label': 'NEUTRAL', 'score': 0.03898626193404198},
# {'label': 'ENTAILMENT', 'score': 0.9592796564102173}
# ]]
