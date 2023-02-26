import json
import math
import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz)), diagonal=1) == 0).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask.unsqueeze(0).unsqueeze(0)

def generate_padding_mask(src, tgt):
    src_padding_mask = (src == 0).transpose(0, 1)
    tgt_padding_mask = (tgt == 0).transpose(0, 1)

    memory_padding_mask = torch.zeros_like(src_padding_mask)

    return src_padding_mask, tgt_padding_mask, memory_padding_mask.transpose(0, 1)

class MusicDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        seq = torch.tensor(seq)
        return seq

def collate_fn(batch):
    sorted_batch = sorted(batch, key=lambda x: len(x), reverse=True)
    sorted_lengths = torch.tensor([len(seq) for seq in sorted_batch])
    padded_batch = pad_sequence(sorted_batch, batch_first=True, padding_value=0)

    return padded_batch, sorted_lengths

class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, src_vocab_size, tgt_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers):
        super(Transformer, self).__init__()

        self.encoder = nn.Embedding(src_vocab_size, d_model)
        self.decoder = nn.Embedding(tgt_vocab_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=hidden_dim, activation="relu")
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers, norm=encoder_norm)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=hidden_dim, activation="relu")
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder_transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers, norm=decoder_norm)

        self.linear = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
        src_emb = self.encoder(src)
        tgt_emb = self.decoder(tgt)

        memory = self.encoder_transformer(src_emb, src_key_padding_mask=src_key_padding_mask.transpose(0, 1))
        output = self.decoder_transformer(tgt_emb, memory, tgt_key_padding_mask=tgt_key_padding_mask)

        output = self.linear(output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.log(torch.tensor(10000.0)) * (torch.arange(0, d_model, 2) / d_model))
        div_term = div_term.unsqueeze(0)

        pe[:, 0::2] = torch.sin(position * div_term) / math.sqrt(d_model)
        pe[:, 1::2] = torch.cos(position * div_term) / math.sqrt(d_model)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
        encoder_output = self.encoder(src, src_key_padding_mask=src_key_padding_mask.transpose(0, 1))
        decoder_output = self.decoder(tgt, encoder_output, tgt_key_padding_mask=tgt_key_padding_mask)
        return decoder_output

src_vocab_size = 4768
tgt_vocab_size = 512
batch_size = 2
d_model = 256
nhead = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 1024
dropout = 0.1
lr = 0.001
num_epochs = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('wordint_data.json', 'r') as f:
    data = json.load(f)

train_data = []
for d in data:
    train_data.append(list(map(int, d)))

train_dataset = MusicDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

model = Transformer(input_dim=src_vocab_size, hidden_dim=dim_feedforward, num_layers=num_encoder_layers,
                    src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size, d_model=d_model,
                    nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers).to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (data, lengths) in enumerate(train_loader, 0):
        src = data[:, :-1].to(device)
        tgt = data[:, 1:].to(device)

        src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask = generate_padding_mask(src, tgt)

        SOS_TOKEN = 0
        src_lengths = torch.LongTensor([len(seq) for seq in src]).to(device)
        tgt_input = torch.cat([torch.tensor([SOS_TOKEN]*tgt.shape[0]).unsqueeze(1).to(device), tgt[:,:-1]], dim=1)

        packed_src = pack_sequence([seq[:length] for seq, length in zip(src, lengths)], enforce_sorted=False)
        packed_tgt = pack_sequence([seq[:length] for seq, length in zip(tgt, lengths)], enforce_sorted=False)

        optimizer.zero_grad()

        output = model(src, tgt_input, src_key_padding_mask=src_key_padding_mask.transpose(0, 1),
                       tgt_key_padding_mask=tgt_key_padding_mask)

        padded_output, _ = pad_packed_sequence(output, batch_first=True, padding_value=0)

        loss = criterion(padded_output.view(-1, tgt_vocab_size), tgt.view(-1))
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')

save_dir = 'saved_models'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'music_transformer_model.pt')
torch.save(model.state_dict(), save_path)