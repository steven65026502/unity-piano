import json
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import device
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import List, Tuple
from PianoServer import data
from PianoServer import dictionary_roll
from memory_profiler import profile

# Hyperparameters
num_epochs = 10
batch_size = 64
learning_rate = 0.001
hidden_size = 256
num_layers = 4
num_heads = 8
feed_forward_size = 512
dropout = 0.1
print_every = 100
music_length = 32

# Define MusicTransformer model
class MusicTransformer(nn.Module):

    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int, num_heads: int, feed_forward_size: int, dropout: float):
        super(MusicTransformer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, feed_forward_size, dropout)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for block in self.transformer_blocks:
            x, hidden = block(x, hidden)
        x = self.output_layer(x)
        return x, hidden

# Define TransformerBlock
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, feed_forward_size: int, dropout: float):
        super(TransformerBlock, self).__init__()

        self.multihead_attention = MultiheadAttention(hidden_size, num_heads)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, feed_forward_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward_size, hidden_size),
            nn.Dropout(dropout)
        )
        self.layer_norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, hidden=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Multi-head attention
        x, hidden = self.multihead_attention(x, hidden)
        x = x + x

        # Layer normalization and residual connection
        x_norm = self.layer_norm1(x)
        x = x + self.feed_forward(x_norm)

        # Layer normalization and residual connection
        x_norm = self.layer_norm2(x)
        x = x + x_norm

        return x, hidden

# Define MultiheadAttention
class MultiheadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super(MultiheadAttention, self).__init__()

        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        self.q_layer = nn.Linear(hidden_size, hidden_size)
        self.k_layer = nn.Linear(hidden_size, hidden_size)
        self.v_layer = nn.Linear(hidden_size, hidden_size)

        self.output_layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, hidden=None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = x.shape

        # Reshape input to (batch_size * num_heads, seq_len, head_size)
        x = x.view(batch_size, seq_len, self.num_heads, self.head_size).permute(0, 2, 1, 3)

        # Linear mapping
        q = self.q_layer(x)
        k = self.k_layer(x)
        v = self.v_layer(x)

        # Split into multiple heads
        q = q.view(batch_size, seq_len, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_size).permute(0, 2, 1, 3)

        # Calculate attention weights
        if hidden is not None:
            attn_weights = torch.matmul(q, k.permute(0, 1, 3, 2)) / np.sqrt(self.head_size)
            attn_weights += torch.matmul(hidden, k.permute(0, 1, 3, 2)) / np.sqrt(self.head_size)
        else:
            attn_weights = torch.matmul(q, k.permute(0, 1, 3, 2)) / np.sqrt(self.head_size)

        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply attention weights
        x = torch.matmul(attn_weights, v)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.hidden_size)

        # Linear mapping
        x = self.output_layer(x)

        return x, attn_weights

# Define PositionalEncoding
class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size: int, max_len: int = music_length):
        super(PositionalEncoding, self).__init__()

        self.hidden_size = hidden_size

        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-np.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * np.sqrt(self.hidden_size)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x

# Load training data
with open('wordint_data.json', 'r') as f:
    data.wordint_data = json.load(f)

# Convert training data to PyTorch tensors
inputs = []
targets = []
for seq in data.wordint_data:
    for i in range(len(seq) - music_length):
        inputs.append(seq[i:i+music_length])
        targets.append(seq[i+music_length])
inputs = torch.LongTensor(inputs)
targets = torch.LongTensor(targets)

# Define training dataset
dataset = TensorDataset(inputs, targets)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create model and optimizer
model = MusicTransformer(len(data.wordint_data), hidden_size, num_layers, num_heads, feed_forward_size, dropout)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Define loss function
criterion = nn.CrossEntropyLoss()

# Train model
for epoch in range(num_epochs):
    epoch_loss = 0
    for i, (inputs, targets) in enumerate(dataloader):
        # Send data to GPU
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs.view(-1, len(data.wordint_data)), targets.view(-1))

        # Backward pass and update parameters
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {epoch_loss/len(dataloader):.5f}")

# Save model
torch.save(model.state_dict(), 'music_transformer.pth')

# Use model to generate music
model.eval()
with torch.no_grad():
    # Randomly choose a starting sequence
    seq = data.wordint_data[random.randint(0, len(data.wordint_data)-1)][:music_length]
    generated_seq = seq.copy()

    # Generate new notes until reaching desired length
    for i in range(music_length, music_length+generated_length):
        inputs = torch.LongTensor([seq]).to(device)

        # Generate next note
        outputs = model(inputs)
        _, topi = outputs.topk(1)
        generated_note = topi.item()

        # Add generated note to sequence
        generated_seq.append(generated_note)

        # Shift sequence one position forward
        seq = seq[1:] + [generated_note]

    # Convert generated notes back to word form
    generated_word_seq = dictionary_roll.inttoword(generated_seq)

    # Write generated notes to file
    with open('generated_music.json', 'w') as f:
        json.dump(generated_word_seq, f)
