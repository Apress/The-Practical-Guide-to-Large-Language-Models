from transformers import pipeline
from bs4 import BeautifulSoup
import requests

generator = pipeline(
    "text-generation",
    # Model size: 954M
    model = "Qwen/Qwen2.5-0.5B-Instruct"
)

question = "When does the action of the movie Alien: Romulus take place?"

# Article about Alien: Romulus movie
alien_romulus_url = "https://avp.fandom.com/wiki/Alien:_Romulus"

# parsing the webpage to get the text
response = requests.get(alien_romulus_url)
soup = BeautifulSoup(response.text, 'html.parser')
context = ' '.join([p.text for p in soup.find_all('p')])

prompt_with_context =\
    f"Use the context below to answer the question. \n"\
    f"Context: {context} \n"\
    f"Question: {question} \n"\
    f"Answer:"

prompt_without_context =\
    f"Question: {question} \n"\
    f"Answer:"

response_with_context = generator(
    prompt_with_context,
    max_new_tokens = 200,
    # Greedy decoding (temperature = 0.0)
    do_sample = False
)

response_without_context = generator(
    prompt_without_context,
    max_new_tokens = 200
)

context_generated_text = response_with_context[0]["generated_text"]
vanilla_generated_text = response_without_context[0]["generated_text"]

context_response = context_generated_text.split("\nAnswer:")[1]
vanilla_response = vanilla_generated_text.split("\nAnswer:")[1]

print("Response with context:")
print(context_response)
print('====================')

print("Response without context:")
print(vanilla_response)
print('====================')

# Response with context:
#  2142[9] between the events of Alien and Aliens.

# Response without context:
#  The film takes place in the year 2100, in a future society where humans have evolved to form large, multi-planet
#  colonies. In this setting, the story is set in an alien planet called Romulus, which has been colonized by human
#  settlers.
