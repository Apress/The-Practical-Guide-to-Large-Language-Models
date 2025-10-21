# Import necessary libraries
from torchview import draw_graph
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda")\
    if torch.cuda.is_available()\
    else torch.device("cpu")

# Let's see the graph of Facebook's RoBERTa model
model_name = "roberta-large-mnli"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# To visualize the model, we need to provide an example input
# It can be absolutely anything that matches the model input format
# You don't need to use real data
# Here we use a premise and hypothesis for a natural language inference task
premise = "Sun is shining today in Berlin"
hypothesis = "The weather is sunny in Berlin"

# Tokenize the input text
input = tokenizer(
    premise,
    hypothesis,
    truncation = True,
    return_tensors = "pt"
)
x = input["input_ids"].to(device)

# Draw the model graph
model_graph = draw_graph(
    model = model,
    input_data = x,
    # Depth of the graph (the higher the value, the more detailed the graph)
    depth = 3,
    # Shows the internals of nested models
    expand_nested = True,
    # Hides intermediate tensor computations
    hide_inner_tensors = True,
    # The name of the graph
    graph_name = 'RoBERTa Model Graph'
)

# View the graph
model_graph.visual_graph.view()
