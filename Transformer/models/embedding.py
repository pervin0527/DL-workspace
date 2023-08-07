import math
import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.emb(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000, drop_prob=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=drop_prob)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0,max_seq_len).unsqueeze(1)
        base = torch.ones(d_model//2).fill_(10000)
        pow_term = torch.arange(0, d_model, 2) / torch.tensor(d_model,dtype=torch.float32)
        div_term = torch.pow(base,pow_term)

        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)

        pe = pe.unsqueeze(0)

        # pe를 학습되지 않는 변수로 등록
        self.register_buffer('positional_encoding', pe)

    def forward(self, x):
        x = x + torch.autograd.Variable(self.positional_encoding[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)