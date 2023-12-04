import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_embed, max_seq_len=256, device=torch.device("cpu")):
        super(PositionalEncoding, self).__init__()

        ## positional encoding 결과를 담을 텐서. -> [max_seq_len, d_embed]
        encoding = torch.zeros(max_seq_len, d_embed, requires_grad=False)

        ## 0부터 max_seq_len-1까지의 연속된 정수값을 포함하는 1차원 텐서. -> [max_seq_len, 1]
        position = torch.arange(0, max_seq_len).float().unsqueeze(1)
        
        ## 10000**(2i/d_embed) == e**(log(10000) * 2i / d_embed)
        div_term = torch.exp(torch.arange(0, d_embed, 2) * -(math.log(10000.0) / d_embed)) 

        ## i가 홀수일 때는 cos함수, j가 짝수일 때는 sin함수.
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)

        ## [1, max_seq_len, d_embed]
        self.encoding = encoding.unsqueeze(0).to(device) 

    def forward(self, x):
        ## x : Word Embedded Vector
        _, seq_len, _ = x.size() ## [batch_size, max_seq_len, d_embed]
        pos_embed = self.encoding[:, :seq_len, :] ## slicing [1, max_seq_len, d_embed]

        out = x + pos_embed ## word embedded vector에 positional encoding값을 더함.

        return out
