import math
import os
import json
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from dictionary_roll import onseteventstoword, wordtoint

# 超参数
vocab_size = 4768
max_len = 512
batch_size = 8
d_model = 256
nhead = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 1024
dropout = 0.1
lr = 0.001
num_epochs = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        
        # Initialize positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe[:self.max_len, :]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to input
        seq_len = x.size(1)
        pe = self.pe[:seq_len, :]
        pe = pe.unsqueeze(0).repeat(x.size(0), 1, 1)[:, :seq_len, :]
        x = x + pe

        # Apply dropout
        x = self.dropout(x)
        return x


class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super(MusicTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout)
        self.decoder = nn.Linear(d_model, vocab_size)
    
    def forward(self, src, tgt):
    # Embedding
        src_embedded = self.embedding(src)
        tgt_embedded = self.embedding(tgt)


    # Positional encoding
        src_embedded = self.pos_encoder(src_embedded)
        tgt_embedded = self.pos_encoder(tgt_embedded)

    # Transformer
        src_mask = self.transformer.generate_square_subsequent_mask(src.shape[1]).to(device)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[1]).to(device)
        memory = self.transformer.encoder(src_embedded.transpose(0, 1), src_key_padding_mask=src == 0, mask=src_mask)
        output = self.transformer.decoder(tgt_embedded.transpose(0, 1), memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt == 0, memory_key_padding_mask=src == 0)

    # Linear layer
        output = self.decoder(output.transpose(0, 1))
        return output   

import glob
data_path = "C:/Users/z7913/Documents/project/unity-piano/pianoroll"

class WordIntDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.word2int = {}
        self.int2word = {}
        self.tokens = []
        json_data = {"key1": "value1", "key2": "value2"}
        for json_data in self.data:
            # 将字典转换为整数列表
            words = onseteventstoword(json_data)
            seq = wordtoint(words)
            self.tokens.extend(seq)

        # 构建词表
        for token in self.tokens:
            if token not in self.word2int:
                index = len(self.word2int)
                self.word2int[token] = index
                self.int2word[index] = token
        # 添加 <EOS> 到词表
        self.word2int["<EOS>"] = len(self.word2int)
        self.int2word[len(self.int2word)] = "<EOS>"
        # 添加 <PAD> 到词表
        self.word2int["<PAD>"] = len(self.word2int)
        self.int2word[len(self.int2word)] = "<PAD>"

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        seq_len = max_len
        seq = self.tokens[index : index + seq_len]
        if len(seq) < seq_len:
            seq += [self.word2int["<PAD>"]] * (seq_len - len(seq))
        src = torch.tensor(seq)
        tgt_input = src.clone()
        tgt_output = src.clone()
        tgt_output = tgt_output.unsqueeze(1)
        tgt_output[:, -1] = self.word2int["<EOS>"]
        if src.shape[0] < max_len:
            pad_tensor = torch.tensor([self.word2int["<PAD>"]] * (max_len - src.shape[0]))
            src = torch.cat((src, pad_tensor))
            tgt_input = torch.cat((tgt_input, pad_tensor))
            tgt_output = torch.cat((tgt_output, torch.tensor([[self.word2int["<PAD>"]] * max_len])), dim=1)
        return src, tgt_input, tgt_output, len(seq)



# 加载数据
train_data = []
for json_file in glob.glob(data_path + '/*.json'):
    with open(json_file, 'r') as f:
        json_data = json.load(f)
        # 进行需要的额外处理
        train_data.append(json_data)

train_dataset = WordIntDataset(train_data)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#初始化模型
model = MusicTransformer(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.word2int["<PAD>"])

# 将数据集分成新的训练集和验证集
val_ratio = 0.2
num_val_samples = int(len(train_dataset) * val_ratio)
num_train_samples = len(train_dataset) - num_val_samples
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [num_train_samples, num_val_samples])

# 初始化数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 训练模型
for epoch in range(num_epochs):
    # 训练
    model.train()
    for i, batch in enumerate(train_dataloader):
        src, tgt, tgt_out, lengths = [x.to(device) for x in batch]
        optimizer.zero_grad()
        output = model(src, tgt)
        loss = criterion(output.reshape(-1, output.shape[2]), tgt_out.reshape(-1))
        loss.backward()
        if torch.isnan(loss):
            continue
        optimizer.step()
        if i % 10 == 0:
            print(f"Epoch {epoch + 1}, Batch {i + 1}, Training Loss: {loss.item():.4f}")

    # 验证
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            src, tgt, tgt_out, lengths = [x.to(device) for x in batch]
            output = model(src, tgt)
            loss = criterion(output.reshape(-1, output.shape[2]), tgt_out.reshape(-1))
            total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.4f}")

    # 重新加载数据集
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

torch.save(model.state_dict(), 'model.pt')