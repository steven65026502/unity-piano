import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch import device

data_path = 'C:\\Users\\謝向嶸\\Desktop\\專題用\\PIANO\\pianoroll\\0.json'

# 读取处理后的数据
with open(data_path, 'r') as f:
    processed_notes = f.read()

# 将字符串转换为列表并转换为NumPy数组
notes = np.array(processed_notes.split(' '))

# 构建字典，将音符映射到整数值
note_to_int_processed = {note: i for i, note in enumerate(set(notes))}
encoded_processed_notes = np.array([note_to_int_processed[note] for note in notes])

# 定义模型参数
input_size = output_size = len(note_to_int_processed)
hidden_size = 256
num_layers = 4
num_heads = 8
dim_feedforward = 512
dropout = 0.2
batch_size = 64
seq_length = 32
learning_rate = 0.001
num_epochs = 100

# 创建训练用的数据集
def create_dataset(seq_length, encoded_notes):
    X = []
    y = []
    for i in range(0, len(encoded_notes) - seq_length):
        X.append(encoded_notes[i:i + seq_length])
        y.append(encoded_notes[i + seq_length])
    X = np.array(X)
    y = np.array(y)
    return X, y

X, y = create_dataset(seq_length, encoded_processed_notes)
dataset = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 定义模型
class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = nn.Sequential(
            nn.Linear(ninp, nhid),
            nn.Tanh(),
            nn.Linear(nhid, nhead)
        )
        encoder_layers = TransformerEncoderLayer(nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * np.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask ==0, float('-inf')).masked_fill(mask == 1, 0.0)


# 创建音符到数字的映射
note_to_int = dict()
int_to_note = dict()
for i, note in enumerate(set(notes)):
    note = note.strip()
    note_to_int[note] = i
    int_to_note[i] = note

# 将音符转换为整数编码
encoded_notes = [note_to_int[note.strip()] for note in notes]

# 划分训练集和测试集
train_ratio = 0.8
train_size = int(len(encoded_notes) * train_ratio)
train_set = encoded_notes[:train_size]
test_set = encoded_notes[train_size:]


# 定义模型
class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = nn.Sequential(
            nn.Linear(ninp, nhid),
            nn.Tanh(),
            nn.Linear(nhid, nhead)
        )
        encoder_layers = TransformerEncoderLayer(nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, mask=None):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * np.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


# 定义训练函数
def train(model, train_loader, criterion, optimizer, device):
    """
    Trains the given model on the training set using the provided optimizer and loss criterion.

    Args:
        model: The model to train.
        train_loader: A PyTorch DataLoader for the training set.
        criterion: The loss criterion to use (e.g. nn.CrossEntropyLoss).
        optimizer: The optimizer to use (e.g. torch.optim.Adam).
        device: The device to run the training on (e.g. 'cpu' or 'cuda').

    Returns:
        The total loss and accuracy over the entire training set.
    """

    # Set the model to train mode
    model.train()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    for i, (inputs, targets) in enumerate(train_loader):
        # Move the data to the device
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Compute the accuracy and update the running totals
        _, predictions = torch.max(outputs, dim=1)
        total_correct += (predictions == targets).sum().item()
        total_samples += targets.size(0)
        total_loss += loss.item()

    # Compute the average loss and accuracy over the entire training set
    avg_loss = total_loss / len(train_loader)
    avg_accuracy = total_correct / total_samples

    return avg_loss, avg_accuracy

def evaluate(model, test_data, batch_size, criterion):
    model.eval()
    test_loss = 0
    n_batches = len(test_data) // batch_size
    with torch.no_grad():
        for i in range(n_batches):
            batch_x = test_data[i*batch_size:(i+1)*batch_size]
            batch_y = batch_x[1:]
            batch_x = batch_x[:-1]
            x = batch_x.T
            y = batch_y
            output = model(x, None)
            loss = criterion(output.view(-1, output.shape[-1]), y)
            test_loss += loss.item()
    return test_loss / n_batches
def test(model, test_loader, criterion):
    model.eval()  # 将模型切换到评估模式
    test_loss = 0.0
    test_size = len(test_loader.dataset)  # 获取测试集大小
    with torch.no_grad():  # 关闭梯度计算
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)  # 将损失值累加到总损失中

    test_loss /= test_size  # 计算平均测试误差
    return test_loss


