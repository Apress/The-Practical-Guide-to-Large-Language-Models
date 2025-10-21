# Importing necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# A tiny attention module
class TinyAttention(nn.Module):

    def __init__(self, d_m = 4):
        """d_m - dimension of model (embedding size)"""
        super().__init__()
        # Q - query, K - key, V - value
        self.q_proj = nn.Linear(d_m, d_m, bias = False)
        self.k_proj = nn.Linear(d_m, d_m, bias = False)
        self.v_proj = nn.Linear(d_m, d_m, bias = False)

    def forward(self, x):
        # x: (B, T, D)
        Q = self.q_proj(x)  # (B, T, D)
        K = self.k_proj(x)
        V = self.v_proj(x)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(x.size(-1))  # (B, T, T)
        # Attention weights
        attn = F.softmax(scores, dim = -1)
        # Weighted sum of values
        out = attn @ V  # (B, T, D)
        return out


# Embeddings for two tokens
w_man = torch.tensor([1.0, 0.0, 0.0, 0.0])
w_bites = torch.tensor([0.0, 1.0, 0.0, 0.0])
w_dog = torch.tensor([0.0, 0.0, 1.0, 0.0])

# Two sentences with the same tokens in different orders
sent1 = torch.stack([w_man, w_bites, w_dog]).unsqueeze(0)
sent2 = torch.stack([w_dog, w_bites, w_man]).unsqueeze(0)

attn = TinyAttention(d_m = 4)

# Get the outputs
out1 = attn(sent1)
out2 = attn(sent2)

print("Output for 'man bites dog':")
print(out1)

print("Output for 'dog bites man':")
print(out2)
