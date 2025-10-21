import gradio as gr
from transformers import pipeline

# Loading pipeline for sentiment analysis
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model = "distilbert-base-uncased-finetuned-sst-2-english"
)


# Sentiment analysis function for Gradio
def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    label = result['label']
    score = result['score']
    return f"Label: {label}, Confidence: {score:.2f}"


# Gradio interface
demo = gr.Interface(
    fn = analyze_sentiment,
    inputs = gr.Textbox(lines = 3, placeholder = "Enter text here..."),
    outputs = "text",
    title = "Sentiment Analyzer with DistilBERT",
    description = "Enter an English sentence and get its sentiment (positive/negative)."
)

# Launching the Gradio app
demo.launch(share = True)
