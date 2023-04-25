
import torch
from hparams import device
from dictionary_roll import pad_token 

def create_padding_mask(inp, n=4):

    mask = torch.eq(inp, pad_token).float()

    return mask.view(*mask.shape[:-1], *[1 for _ in range(n-2)], mask.shape[-1]).to(inp.device)

def create_look_ahead_mask(seq_len):

    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask.float().to(device)

def create_mask(inp, n=4):
    # padding mask
    padding_mask = create_padding_mask(inp, n=n)
    # look ahead mask, assuming seq_len is last dimension of inp
    look_ahead_mask = create_look_ahead_mask(inp.shape[-1])
    # final mask is the maximum of the two
    combined_mask = torch.max(padding_mask, look_ahead_mask)
    return combined_mask
