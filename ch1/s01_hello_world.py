from transformers import pipeline

import os

os.environ['HF_HOME'] = "~/.cache/huggingface/hub"

classifier = pipeline(
    "sentiment-analysis",
    # Model size: 256M
    model = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

response = classifier("I like Large Language Models very much!")

print(response)

# [{
# 'label': 'POSITIVE',
# 'score': 0.9997369647026062
# }]
