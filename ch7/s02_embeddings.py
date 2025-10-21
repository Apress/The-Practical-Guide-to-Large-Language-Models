# Import necessary libraries
import numpy as np
from gensim.matutils import unitvec
from gensim.models import KeyedVectors
import gensim.downloader as api

# Download the pre-trained Word2Vec model (Google News vectors)
print("Downloading word2vec-google-news-300 via gensim.downloader ...")
# Downloads ~1.5 GB to cache and loads the model
w2v: KeyedVectors = api.load("word2vec-google-news-300")


# Function to get the embedding of a word and normalize it
def emb(word: str) -> np.ndarray:
    return unitvec(w2v[word])


# Cosine similarity between two vectors
def cos(a, b) -> float:
    return float(np.dot(a, b))


# Comparing (man + throne) embedding with king and sea
man_throne_vec = unitvec(emb("man") + emb("throne"))
print(cos(man_throne_vec, emb("king")))
# Returns: 0.5140514373779297
print(cos(man_throne_vec, emb("sea")))
# Returns: 0.11857952922582626

# Comparing (woman + throne) embedding with queen and sea
woman_throne_vec = unitvec(emb("woman") + emb("throne"))
print(cos(woman_throne_vec, emb("queen")))
# Returns: 0.5087150931358337
print(cos(woman_throne_vec, emb("sea")))
# Returns: 0.12249107658863068
