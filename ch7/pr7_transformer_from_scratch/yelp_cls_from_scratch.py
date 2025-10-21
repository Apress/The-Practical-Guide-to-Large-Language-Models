# Necessary imports
import math
import os
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


# This section defines a lightweight Dataset wrapper for the Yelp Polarity corpus.
# It stores raw texts and integer labels, and exposes the minimal Dataset API
# (__len__ and __getitem__) needed by a DataLoader.
# Yelp Review Dataset Wrapper
class YelpTextDataset(Dataset):

    def __init__(self, texts: List[str], labels: List[int]):
        assert len(texts) == len(labels)
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], int(self.labels[idx])


# The collator converts a batch of variable-length strings into fixed-size tensors.
# It tokenizes texts, builds padding masks, and (crucially here) looks up static
# token embeddings from a frozen embedding matrix instead of a trainable embedding layer.
# Collator
def make_collate_with_static_embeddings(
        tokenizer: AutoTokenizer,
        # FROZEN embedding weights for token lookup
        # (vocab_size, hidden)
        emb_weight: torch.Tensor,
        max_len: int,
):
    """
    We return a closure ("collate") that the DataLoader will call on each batch.
    The closure has access to the tokenizer, frozen embedding weights, and max_len.
    This lets us keep the model focused on learning Transformer parameters only.
    """

    # The collate function turns a list of (text, label) into model-ready tensors:
    # token ids, attention masks, padding masks, and precomputed token embeddings.
    # Using embeddings here eliminates the need for an embedding layer in the model.
    def collate(batch: List[Tuple[str, int]]):
        texts, labels = zip(*batch)
        enc = tokenizer(
            list(texts),
            padding = "max_length",
            truncation = True,
            max_length = max_len,
            return_tensors = "pt",
        )

        # Token indices for each position (B = batch, T = sequence length).
        input_ids = enc["input_ids"]  # (B, T)

        # 1 where token is real, 0 where it's padding. Used for masking and pooling.
        attention_mask = enc["attention_mask"]  # (B, T)

        # Boolean mask: True at padding positions. Suitable for attention masking.
        pad_mask = attention_mask.eq(0)  # (B, T)

        # ---- FIXED VECTORIZATION: simple token lookup with no context ----
        # emb_weight: (V, D); input_ids: (B, T) -> x_emb: (B, T, D)
        # We deliberately use a pure embedding lookup—no positions, no self-attn, no training here.
        x_emb = F.embedding(input_ids, emb_weight)

        # Convert labels to a LongTensor (required by CrossEntropyLoss).
        labels_t = torch.tensor(labels, dtype = torch.long)

        # Return a dict matching the model's expected inputs.
        return {
            "embeddings":     x_emb,  # (B, T, D) — ready for the model
            "attention_mask": attention_mask,  # (B, T)
            "pad_mask":       pad_mask,  # (B, T)
            "labels":         labels_t,  # (B,)
        }

    return collate


# Sinusoidal Positional Encoding
# Classic "Attention Is All You Need" fixed positional encoding.
# Adds deterministic sin/cos signals so the model can infer token order
# even when token embeddings themselves are position-agnostic.
class SinusoidalPositionalEncoding(nn.Module):

    def __init__(
            self,
            # d_model is the hidden size per token; positions will produce vectors of this size.
            d_model: int,
            max_len: int = 5000
    ):
        super().__init__()

        # Precompute a (max_len, d_model) table of sin/cos values once at init time.
        pe = torch.zeros(max_len, d_model)

        # Positions 0..max_len-1 as a column vector to broadcast with frequencies.
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)

        # Geometric progression of frequencies; even dims get sin, odd dims get cos.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even dimensions.
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd dimensions.
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension so we can slice by sequence length during forward.
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        # Register as a buffer so it moves with the module across devices, but isn't a parameter.
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        # x: (B, T, D)
        return x + self.pe[:, :x.size(1)]


# Multi-Head Self-Attention layer: projects inputs to Q/K/V, computes scaled dot-product
# attention per head with optional padding mask, then recombines heads and projects out.
# Model
class MultiHeadSelfAttention(nn.Module):

    def __init__(
            self,
            d_model: int,
            n_heads: int = 3,
            dropout: float = 0.1
    ):
        super().__init__()

        # Each head must have an integer dimension; head_dim = d_model / n_heads.
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Linear projections for queries (Q).
        self.q_proj = nn.Linear(d_model, d_model)

        # Linear projections for keys (K).
        self.k_proj = nn.Linear(d_model, d_model)

        # Linear projections for values (V).
        self.v_proj = nn.Linear(d_model, d_model)

        # Final linear layer after concatenating all heads.
        self.out_proj = nn.Linear(d_model, d_model)

        # Dropout on attention weights to regularize.
        self.dropout = nn.Dropout(dropout)

    # x: (B, T, D). pad_mask: (B, T) with True at padding positions to be masked.
    def forward(
            self,
            x: torch.Tensor,
            pad_mask: Optional[torch.Tensor] = None
    ):

        # B: batch size, T: sequence length. D equals d_model.
        B, T, _ = x.size()

        # Project inputs to queries.
        q = self.q_proj(x)

        # Project inputs to keys.
        k = self.k_proj(x)

        # Project inputs to values.
        v = self.v_proj(x)

        # Reshape (B, T, D) -> (B, n_heads, T, head_dim) and put head before time.
        def reshape_heads(t):

            # View splits features across heads, permute brings head dim forward.
            return t.view(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Split Q/K/V across heads.
        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        # Scaled dot-product attention: scores over keys for each query position.
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if pad_mask is not None:
            # pad_mask: (B, T) -> (B, 1, 1, T)
            mask = pad_mask.unsqueeze(1).unsqueeze(2)

            # Fill scores at padding positions with -inf so softmax gives them zero weight.
            scores = scores.masked_fill(mask, float("-inf"))

        # Normalize to probabilities over key positions.
        attn = torch.softmax(scores, dim = -1)

        # Regularize attention distribution.
        attn = self.dropout(attn)

        # Weighted sum of values using attention weights.
        context = torch.matmul(attn, v)

        # Recombine heads: (B, n_heads, T, head_dim) -> (B, T, D).
        context = context.permute(0, 2, 1, 3).contiguous().view(B, T, self.d_model)

        # Final linear to mix head outputs.
        out = self.out_proj(context)

        # Return shape (B, T, D), same as input embedding shape.
        return out


# A standard Transformer encoder block: Pre-LN -> MHA (+residual) -> Pre-LN -> FFN (+residual).
# LayerNorm before sublayers often stabilizes training vs. the original Post-LN variant.
class TransformerEncoderLayer(nn.Module):

    def __init__(
            self,
            # Model hidden size; dimensionality of token vectors throughout the block.
            d_model: int,
            # Number of parallel attention heads.
            n_heads: int = 3,
            # Feed-forward expansion size (typically 4x d_model).
            d_ff: int = 4096,
            # Dropout rate applied in attention and FFN.
            dropout: float = 0.1
    ):
        super().__init__()

        # Pre-norm before attention.
        self.ln1 = nn.LayerNorm(d_model)

        # Multi-head self-attention sublayer.
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)

        # Dropout on the residual branch after attention.
        self.dropout1 = nn.Dropout(dropout)

        # Pre-norm before feed-forward network.
        self.ln2 = nn.LayerNorm(d_model)

        # Position-wise FFN with GELU nonlinearity and dropout in between.
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        # Dropout on the residual branch after FFN.
        self.dropout2 = nn.Dropout(dropout)

    # pad_mask masks out padding tokens inside attention.
    def forward(
            self,
            x: torch.Tensor,
            pad_mask: Optional[torch.Tensor] = None
    ):
        # Normalize then apply attention.
        h = self.ln1(x)

        # Self-attention over the sequence (contextualization).
        h = self.attn(h, pad_mask)

        # Residual connection from input to attention output.
        x = x + self.dropout1(h)

        # Normalize then apply FFN.
        h2 = self.ln2(x)

        # Nonlinear mixing of features per position.
        h2 = self.ffn(h2)

        # Residual connection from input to FFN output.
        x = x + self.dropout2(h2)

        # Return the updated sequence representation.
        return x


# A minimal Transformer-based text classifier built on frozen token embeddings.
# It adds sinusoidal positions, stacks encoder layers, pools with a masked mean,
# and predicts class logits via a linear head.
class TransformerClassifier(nn.Module):

    def __init__(
            self,
            # Hidden size of token embeddings (must match frozen embedding dim).
            d_model: int,
            # Number of attention heads in each encoder layer.
            n_heads: int = 3,
            # Number of stacked encoder layers.
            n_layers: int = 3,
            # Maximum sequence length supported by positional encoding.
            max_len: int = 256,
            # Number of sentiment classes (Yelp Polarity has 2).
            num_classes: int = 2,
            # Dropout applied to inputs and inside layers.
            dropout: float = 0.1,
    ):
        super().__init__()
        # Fixed sinusoidal positions; no learned positional embeddings.
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)
        # Input dropout for regularization.
        self.dropout = nn.Dropout(dropout)
        # Stack of identical encoder layers.
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model = d_model,
                n_heads = n_heads,
                d_ff = 4 * d_model,
                dropout = dropout
            )
            for _ in range(n_layers)
        ])

        # Final LayerNorm to stabilize output before pooling/head.
        self.ln_f = nn.LayerNorm(d_model)

        # Linear classifier mapping pooled vector to class logits.
        self.head = nn.Linear(d_model, num_classes)

    # x_emb: frozen token embeddings; attention_mask/pad_mask: masks from tokenizer.
    def forward(
            self,
            x_emb: torch.Tensor,
            attention_mask: torch.Tensor,
            pad_mask: torch.Tensor
    ):
        # x_emb: (B, T, D) — already token vectors (independent)

        # Positional Encoding.
        # Scale by sqrt(D) (common for embeddings) and add sinusoidal positions.
        x = self.pos_enc(x_emb * math.sqrt(self.pos_enc.pe.size(-1)))

        # Apply dropout before entering the encoder stack.
        x = self.dropout(x)

        # Pass through N encoder layers with shared padding mask.
        for layer in self.layers:
            x = layer(x, pad_mask)

        # Normalize features before pooling and classification.
        x = self.ln_f(x)

        # attention_mask: (B, T), 1 — token, 0 — pad. Expand to (B, T, 1) to mask features.
        mask = attention_mask.unsqueeze(-1)  # (B, T, 1)

        # Sum only over real tokens (padding contributes zero).
        summed = (x * mask).sum(dim = 1)  # (B, D)

        # Count of real tokens per example (avoid divide-by-zero with clamp).
        denom = mask.sum(dim = 1).clamp(min = 1)  # (B, 1)

        # Masked mean pooling over time.
        pooled = summed / denom  # (B, D)

        # Final classification layer to logits.
        logits = self.head(pooled)

        # Return unnormalized class scores.
        return logits


# Evaluation
# Evaluate the model on a DataLoader: compute average loss and accuracy without gradients.
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    # Cross-entropy for multi-class classification with integer labels.
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():

        # Iterate over validation batches with a progress bar.
        for batch in tqdm(loader, desc = "Validation", leave = False):

            # Move tensors to device; embeddings already computed in collate.
            x_emb = batch["embeddings"].to(device)  # (B, T, D)
            attention_mask = batch["attention_mask"].to(device)
            pad_mask = batch["pad_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass to get logits.
            logits = model(x_emb, attention_mask, pad_mask)

            # Compute batch loss.
            loss = criterion(logits, labels)

            # Accumulate total loss weighted by batch size for proper averaging.
            loss_sum += loss.item() * x_emb.size(0)

            # Predicted class is argmax over logits.
            preds = logits.argmax(dim = -1)

            # Count correct predictions.
            correct += (preds == labels).sum().item()

            # Track number of evaluated samples.
            total += x_emb.size(0)

    # Return mean loss and accuracy over the entire validation set.
    return loss_sum / max(1, total), correct / max(1, total)


# Training setup and hyperparameters used for this experiment.
train_subset = 50_000
test_subset = 10_000

# Pre-trained sentence transformer model to borrow embeddings from.
st_model_name = "sentence-transformers/all-MiniLM-L6-v2"

# Maximum sequence length for tokenization and model.
max_len = 256

# Max gradient norm for clipping to stabilize training.
max_grad_norm = 1.0

# Mini-batch size for training and evaluation.
batch_size = 64

# Dropout rate inside the model.
dropout = 0.05

# Learning rate for AdamW optimizer.
lr = 2e-4

# Weight decay (L2 regularization) for AdamW.
weight_decay = 0.01

# Where to store the best checkpoint.
out_dir = "/tmp/checkpoints"

# Number of training epochs.
epochs = 5

# Heuristic to drop very long reviews to keep training efficient.
char_limit = max_len * 6

# Load dataset
print("Loading dataset...")
ds = load_dataset("fancyzhx/yelp_polarity")

# Filter out texts exceeding the character limit to bound sequence lengths.
ds = ds.filter(lambda x: len(x["text"]) <= char_limit)

# Split into train/test subsets as provided by the dataset.
train_ds = ds["train"]
test_ds = ds["test"]

# Respect subset caps if specified; otherwise use full splits.
train_size = len(train_ds) if train_subset < 0 else min(train_subset, len(train_ds))
test_size = len(test_ds) if test_subset < 0 else min(test_subset, len(test_ds))

# Materialize the chosen number of examples into Python lists.
train_texts = [train_ds[i]["text"] for i in range(train_size)]
train_labels = [int(train_ds[i]["label"]) for i in range(train_size)]
test_texts = [test_ds[i]["text"] for i in range(test_size)]
test_labels = [int(test_ds[i]["label"]) for i in range(test_size)]

# Load a tokenizer compatible with the embedding model we will borrow.
tokenizer = AutoTokenizer.from_pretrained(st_model_name, use_fast = True)

# Load the backbone model only to extract its input embedding matrix (kept frozen).
backbone = AutoModel.from_pretrained(st_model_name)
with torch.no_grad():

    # Grab the token embedding layer of the backbone.
    tok_emb_layer: nn.Embedding = backbone.get_input_embeddings()

    # emb_weight: dimension (V, D) — a frozen lookup table we’ll reuse in the collator.
    emb_weight = tok_emb_layer.weight.detach().cpu().clone()

# d_model matches the embedding dimension of the borrowed embeddings.
d_model = emb_weight.size(1)

# Build a collator that uses the frozen embedding matrix for token lookup.
collate = make_collate_with_static_embeddings(
    tokenizer = tokenizer,
    # Pass the precomputed (and frozen) embedding weights.
    emb_weight = emb_weight,
    max_len = max_len,
)

# Training DataLoader with shuffling and our custom collate function.
train_loader = DataLoader(
    YelpTextDataset(train_texts, train_labels),
    batch_size = batch_size,
    shuffle = True,
    collate_fn = collate,
    pin_memory = True,
)

# Test DataLoader without shuffling, same collate for consistency.
test_loader = DataLoader(
    YelpTextDataset(test_texts, test_labels),
    batch_size = batch_size,
    shuffle = False,
    collate_fn = collate,
    pin_memory = True,
)

# Choose GPU if available; otherwise fall back to CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Instantiate the Transformer classifier with the chosen depth/width.
model = TransformerClassifier(
    d_model = d_model,
    n_heads = 6,
    n_layers = 3,
    max_len = max_len,
    num_classes = 2,
    dropout = dropout,
).to(device)


# Counting model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Print the number of parameters
num_params = count_parameters(model)
print(f"Model has {num_params:,} trainable parameters")

# Optional: Display the model architecture (torchview)
try:
    from torchview import draw_graph
    import graphviz

    # Move model to CPU for visualization
    model_cpu = model.to("cpu").eval()

    # Create dummy inputs matching the model's expected input shapes.
    B, T, D = 2, max_len, d_model
    x_dummy = torch.randn(B, T, D)  # (B, T, D)
    attention_mask_dummy = torch.ones(B, T, dtype = torch.long)
    # some padding at the end
    attention_mask_dummy[:, -8:] = 0
    pad_mask_dummy = attention_mask_dummy.eq(0)  # (B, T) bool

    # Draw and render the computation graph.
    model_graph = draw_graph(
        model = model_cpu,
        input_data = (x_dummy, attention_mask_dummy, pad_mask_dummy),
        depth = 2,
        expand_nested = False,
        hide_inner_tensors = True,
        hide_module_functions = True,
        graph_name = "TransformerClassifier"
    )
    # View the graph
    model_graph.visual_graph.view()

finally:
    # Return model to the target device for training
    model = model_cpu.to(device).train()

# AdamW optimizer with weight decay (decoupled L2) is standard for Transformers.
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = lr,
    weight_decay = weight_decay
)

# Cross-entropy for two-class classification (logits -> softmax implicitly inside loss).
criterion = nn.CrossEntropyLoss()

# Track the best validation accuracy to save checkpoints.
best_acc = 0.0

# Ensure the checkpoint directory exists.
os.makedirs(out_dir, exist_ok = True)

# Main training loop over epochs.
for epoch in range(1, epochs + 1):
    # Switch to training mode (enables dropout, etc.).
    model.train()

    # Progress bar over the training batches.
    pbar = tqdm(train_loader, desc = f"Epoch {epoch}/{epochs}")

    # Running metrics for the current epoch.
    running_loss = 0.0
    total = 0
    correct = 0

    # Iterate over mini-batches.
    for batch in pbar:

        # Move batch tensors to the target device.
        x_emb = batch["embeddings"].to(device)  # (B, T, D) — already vectors
        attention_mask = batch["attention_mask"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        labels = batch["labels"].to(device)

        # Clear gradients from the previous step.
        optimizer.zero_grad(set_to_none = True)

        # Forward pass to obtain logits.
        logits = model(x_emb, attention_mask, pad_mask)

        # Compute loss for this batch.
        loss = criterion(logits, labels)

        # Backpropagate to compute gradients.
        loss.backward()

        # Clip gradients to prevent exploding gradients.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Update parameters.
        optimizer.step()

        # Accumulate loss weighted by batch size for running average.
        running_loss += loss.item() * x_emb.size(0)

        # Convert logits to hard predictions.
        preds = logits.argmax(dim = -1)

        # Update accuracy counters.
        correct += (preds == labels).sum().item()

        # Update seen sample count.
        total += x_emb.size(0)

        # Show live loss and accuracy on the progress bar.
        pbar.set_postfix({"loss": f"{running_loss / max(1, total):.4f}", "acc": f"{correct / max(1, total):.4f}"})

    # Validate at the end of each epoch.
    val_loss, val_acc = evaluate(model, test_loader, device)

    # Report validation metrics for tracking.
    print(f"\nValidation — epoch {epoch}: loss={val_loss:.4f}, acc={val_acc:.4f}")

    # Save the checkpoint if validation accuracy improves.
    if val_acc > best_acc:
        best_acc = val_acc
        ckpt_path = os.path.join(out_dir, "best_transformer_static_emb.pt")
        torch.save({
            "model_state": model.state_dict(),
            "val_acc":     val_acc,
        }, ckpt_path)

        # Inform the user about the new best model and where it was saved.
        print(f"Saved best model: {ckpt_path} (acc={best_acc:.4f})")

# Final summary of the best observed validation accuracy.
print(f"Best accuracy: {best_acc:.4f}")

# Epoch 1/5: 100%|██████████| 782/782 [03:45<00:00,  3.46it/s, loss=0.2789, acc=0.8782]
# Validation — epoch 1: loss=0.2573, acc=0.8973

# Epoch 2/5: 100%|██████████| 782/782 [03:47<00:00,  3.44it/s, loss=0.2125, acc=0.9133]
# Validation — epoch 2: loss=0.2175, acc=0.9098

# Epoch 3/5: 100%|██████████| 782/782 [03:46<00:00,  3.45it/s, loss=0.1858, acc=0.9258]
# Validation — epoch 3: loss=0.2219, acc=0.9101

# Epoch 4/5: 100%|██████████| 782/782 [03:47<00:00,  3.44it/s, loss=0.1640, acc=0.9354]
# Validation — epoch 4: loss=0.2152, acc=0.9125

# Epoch 5/5: 100%|██████████| 782/782 [03:46<00:00,  3.45it/s, loss=0.1385, acc=0.9461]
# Validation — epoch 5: loss=0.2169, acc=0.9167

# Best accuracy: 0.9167
