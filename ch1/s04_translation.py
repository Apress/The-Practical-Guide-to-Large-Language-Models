from transformers import pipeline

translator = pipeline(
    "translation",
    # Model size: 411M
    model = "Helsinki-NLP/opus-mt-en-zh"
)

text = "I like Large Language Models very much!"
translation = translator(text)

print(translation)
# [{'translation_text': '我非常喜欢大语言模型!'}]
