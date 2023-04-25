import torch
from math import sqrt
from torch import nn
from dictionary_roll import vocab_size
# hparams
from torch import cuda, device as d

# get device
if cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = d(dev)

# default hparams for the model
hparams = {
    "d_model": 256,
    "num_layers": 6,
    "num_heads": 8,
    "d_ff": 512,
    "max_rel_dist": 1024,
    "max_abs_position": 0,
    "bias": True,
    "dropout": 0.1,
    "layernorm_eps": 1e-6,
    "vocab_size": vocab_size
}

# layers
import torch
import torch.nn.functional as F
from torch import nn
from math import sqrt
from hparams import device

def abs_positional_encoding(max_position, d_model, n=3):
    positions = torch.arange(max_position).float().to(device)
    k = torch.arange(d_model).float().to(device)
    coeffs = 1 / torch.pow(10000, 2 * (k // 2) / d_model)
    angles = positions.view(-1, 1) @ coeffs.view(1, -1)
    angles[:, 0::2] = torch.sin(angles[:, 0::2])
    angles[:, 1::2] = torch.cos(angles[:, 1::2])
    return angles.view(*[1 for _ in range(n-2)], max_position, d_model)

def skew(t):
    padded = F.pad(t, [1, 0])
    Srel = padded.reshape(-1, t.shape[-1] + 1, t.shape[-2])
    Srel = Srel[:, 1:]              
    Srel = Srel.reshape(*t.shape)   
    return Srel

def rel_scaled_dot_prod_attention(q, k, v, e=None, mask=None):
    QKt = torch.matmul(q, k.transpose(-1, -2))  

    if e is None:
        Srel = torch.zeros(*q.shape[:-2], q.shape[-2], k.shape[-2], device=q.device)
    else:
        Srel = skew(torch.matmul(q, e.transpose(-1, -2)))  

    dk = sqrt(k.shape[-1])
    scaled_attention_logits = (QKt + Srel) / dk

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    return torch.matmul(F.softmax(scaled_attention_logits, dim=-    1), v)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_rel_dist, bias=True):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.max_rel_dist = max_rel_dist

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible into num_heads heads")

        self.depth = self.d_model // self.num_heads

        self.wq = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.wk = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.wv = nn.Linear(self.d_model, self.d_model, bias=bias)

        self.E = nn.Embedding(self.max_rel_dist, self.d_model)
        self.wo = nn.Linear(self.d_model, self.d_model, bias=True)

    @staticmethod
    def split_heads(x, num_heads, depth=None):
        if depth is None:
            if x.shape[-1] % num_heads != 0:
                raise ValueError("d_model must be divisible into num_heads")
            depth = x.shape[-1] // num_heads

        x = x.view(*x.shape[:-1], num_heads, depth)
        return x.transpose(-2, -3)

    def get_required_embeddings(self, seq_len, max_len=None):
        
        if max_len is None:
            max_len = self.E.num_embeddings

        # required relative position embeddings
        E_dev = self.E.weight.device
        first_emb = self.E(torch.arange(0, 1, device=E_dev)).clone()
        return torch.cat(
            [*[first_emb.clone() for _ in range(max(seq_len - max_len, 0))],
             self.E(torch.arange(max(max_len - seq_len, 0), max_len, device=E_dev))],
            dim=0
        )

    def forward(self, q, k, v, mask=None):
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        seq_len_k = k.shape[-2]
        e = self.get_required_embeddings(seq_len_k, self.max_rel_dist)

        q = self.split_heads(q, self.num_heads, self.depth)
        k = self.split_heads(k, self.num_heads, self.depth)
        v = self.split_heads(v, self.num_heads, self.depth)
        e = self.split_heads(e, self.num_heads, self.depth)

        rel_scaled_attention = rel_scaled_dot_prod_attention(q, k, v, e, mask=mask)

        rel_scaled_attention = rel_scaled_attention.transpose(-2, -3)
        sh = rel_scaled_attention.shape
        return self.wo(rel_scaled_attention.reshape(*sh[:-2], self.d_model))

class PointwiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, bias=True):
        super(PointwiseFFN, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.main = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=bias)
        )

    def forward(self, x):
        return self.main(x)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_rel_dist, bias=True, dropout=0.1, layernorm_eps=1e-6):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_rel_idst = max_rel_dist

        self.mha = MultiHeadAttention(d_model, num_heads, max_rel_dist, bias)
        self.ffn = PointwiseFFN(d_model, d_ff, bias)

        self.layernorm1 = nn.LayerNorm(normalized_shape=d_model, eps=layernorm_eps)
        self.layernorm2 = nn.LayerNorm(normalized_shape=d_model, eps=layernorm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tgt, memory=None, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        attn_out = self.layernorm1(tgt)
        attn_out = self.mha(attn_out, attn_out, attn_out, mask=tgt_mask)
        attn_out = self.dropout1(attn_out)
        attn_out = tgt + attn_out

        ffn_out = self.layernorm2(attn_out)
        ffn_out = self.ffn(ffn_out)
        ffn_out = self.dropout2(ffn_out)
        ffn_out = ffn_out + attn_out

        return ffn_out
# model
class MusicTransformer(nn.Module):
    def __init__(self,
                 d_model=hparams["d_model"],
                 num_layers=hparams["num_layers"],
                 num_heads=hparams["num_heads"],
                 d_ff=hparams["d_ff"],
                 max_rel_dist=hparams["max_rel_dist"],
                 max_abs_position=hparams["max_abs_position"],
                 vocab_size=hparams["vocab_size"],
                 bias=hparams["bias"],
                 dropout=hparams["dropout"],
                 layernorm_eps=hparams["layernorm_eps"]):
        super(MusicTransformer, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_rel_dist = max_rel_dist
        self.max_position = max_abs_position
        self.vocab_size = vocab_size

        self.input_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = abs_positional_encoding(max_abs_position, d_model)
        self.input_dropout = nn.Dropout(dropout)

        self.decoder = nn.TransformerDecoder(
            DecoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_rel_dist=max_rel_dist,
                         bias=bias, dropout=dropout, layernorm_eps=layernorm_eps),
            num_layers=num_layers,
            norm=nn.LayerNorm(normalized_shape=d_model, eps=layernorm_eps)
        )
        self.final = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.input_embedding(x)
        x *= sqrt(self.d_model)

        if self.max_position > 0:
            x += self.positional_encoding[:, :x.shape[-2], :]

        x = self.input_dropout(x)
        x = self.decoder(x, memory=None, tgt_mask=mask)
        return self.final(x)
