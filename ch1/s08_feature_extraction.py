from transformers import pipeline

feature_extractor = pipeline(
    "feature-extraction",
    # Model size: 535M
    model = "facebook/bart-base",
    framework = "pt"
)

text = "I like walking in the park."
# tokens = ['<s>', 'I', ' like', ' walking', ' in', ' the', ' park', '.', '</s>']

embeddings = feature_extractor(text, return_tensors = "pt")

print(embeddings.shape)
# torch.Size([1, 9, 768])
# where 9 is the number of tokens in the input text
