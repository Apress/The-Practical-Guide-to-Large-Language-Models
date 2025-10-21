from transformers import pipeline

generator = pipeline(
    "text-generation",
    model = "microsoft/Phi-3-mini-4k-instruct",
    # Uncomment the line below to run on CPU if you don't have a GPU
    # device = -1
)

question = "What is DeepSeek?"

prompt_without_context =\
    f"Question: {question} \n"\
    f"Answer:"

response = generator(
    prompt_without_context,
    max_new_tokens = 200,
    do_sample = False
)

generated_text = response[0]["generated_text"]

response = generated_text.split("\nAnswer:")[1]

print("Response:")
print(response)

# Response:
#  DeepSeek is a hypothetical advanced AI system designed for deep-sea exploration and data analysis. It would
#  utilize a combination of autonomous underwater vehicles (AUVs), remotely operated vehicles (ROVs),
#  and sophisticated sensors to collect data from the ocean floor. DeepSeek would process this data using machine
#  learning algorithms to identify patterns, map the seabed, and discover new marine species or geological
#  formations. It could also monitor environmental changes and assess the impact of human activities on marine
#  ecosystems.
