import requests
from bs4 import BeautifulSoup


def split_text(text, size = 500, overlap = 100):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        # Move the start index forward, considering overlap
        start += size - overlap
        if end >= len(words):
            break
    return chunks


# Article about Alien: Romulus movie
alien_romulus_url = "https://avp.fandom.com/wiki/Alien:_Romulus"

# parsing the webpage to get the text
response = requests.get(alien_romulus_url)
soup = BeautifulSoup(response.text, 'html.parser')
text = ' '.join([p.text for p in soup.find_all('p')])

# Splitting the context into fixed-size chunks
chunks = split_text(text, size = 500, overlap = 100)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i + 1}:\n{chunk}\n")
    print("-" * 40)
