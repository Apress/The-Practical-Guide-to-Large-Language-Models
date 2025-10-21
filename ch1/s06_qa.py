# pip install bs4
from transformers import pipeline

qa = pipeline(
    'question-answering',
    # Model size: 475M
    model = "deepset/roberta-base-squad2"
)

question = "When the action of the movie Alien: Romulus takes place?"

# Article about Alien: Romulus movie
alien_romulus_url = "https://avp.fandom.com/wiki/Alien:_Romulus"

from bs4 import BeautifulSoup
import requests

# parsing the webpage to get the text
response = requests.get(alien_romulus_url)
soup = BeautifulSoup(response.text, 'html.parser')
context = ' '.join([p.text for p in soup.find_all('p')])

result = qa(
    question = question,
    context = context
)

print(result)
# {'score': 0.46704813838005066, 'start': 11709, 'end': 11713, 'answer': '2142'}
