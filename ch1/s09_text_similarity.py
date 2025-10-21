from transformers import pipeline
import torch

feature_extractor = pipeline(
    "feature-extraction",
    model = "facebook/bart-base"
)

# Define sentence and two candidates
sentence = "Artificial intelligence is transforming the world."
candidates = [
    "Maradona was one of the best football players in history.",
    "Machine Learning affects all areas of life."
]

# Extract embeddings
s_emb = torch.tensor(feature_extractor(sentence)).squeeze(0)

for candidate in candidates:
    # Extract embeddings
    c_emb = torch.tensor(
        feature_extractor(candidate)
    ).squeeze(0)

    # Compute cosine similarity between the averaged embedding vectors.
    # We average the embeddings across the sequence-length dimension (dim=0)
    # to obtain a single vector representation for each sentence,
    # aggregating semantic information from all tokens in the sentence.
    cosine_similarity = torch.nn.functional.cosine_similarity(
        s_emb.mean(dim = 0, keepdim = True),
        c_emb.mean(dim = 0, keepdim = True)
    )

    print("Candidate:", candidate)
    print("Cosine Similarity:", cosine_similarity.item())
    print('-----')

# Candidate: Maradona was one of the best football players in history.
# Cosine Similarity: 0.4624532461166382
# -----
# Candidate: Machine Learning affects all areas of life.
# Cosine Similarity: 0.7117681503295898
# -----
