from transformers import pipeline

summarizer = pipeline(
    "summarization",
    # Model size: 1553M
    model = "facebook/bart-large-cnn"
)

text = "In recent years, there has been a significant surge in the effectiveness of artificial intelligence models. "\
       "Video generators create clips that are difficult to distinguish from real ones, image generation services "\
       "produce logos better than professional designers, and chatbots based on Large Language Models have completely "\
       "revolutionized people's perception of artificial intelligence. Needless to say, the release of OpenAI's "\
       "ChatGPT had a profound impact on the market. Many people were shocked by the capabilities of this model. Many "\
       "began using it to solve everyday tasks, while professionals started delegating their specialized work to this "\
       "service."

summary = summarizer(text, min_length = 30, max_length = 50)

print(summary)
# [{'summary_text':
# "The release of OpenAI's ChatGPT had a profound impact on the market. Many began using it to solve everyday tasks,
# while professionals started delegating their specialized work to this service."
# }]
