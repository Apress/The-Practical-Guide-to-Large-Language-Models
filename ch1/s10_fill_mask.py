from transformers import pipeline

unmasker = pipeline(
    "fill-mask",
    # Model size: 417M
    model = "google-bert/bert-base-uncased"
)

sentence = "Michael Jordan is a professional [MASK] player."

result = unmasker(sentence)

print(result)

# [
# {'score': 0.7602677941322327, 'token': 3455, 'token_str': 'basketball'},
# {'score': 0.09325960278511047, 'token': 3598, 'token_str': 'baseball'},
# {'score': 0.051540181040763855, 'token': 2374, 'token_str': 'football'},
# {'score': 0.044071413576602936, 'token': 5093, 'token_str': 'tennis'},
# {'score': 0.01763438992202282, 'token': 4715, 'token_str': 'soccer'}
# ]
