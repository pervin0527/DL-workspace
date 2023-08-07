import math
import torch
from torch import nn
from torch.nn import functional as F
from models.utils import get_clones

class LayerNorm(nn.Module):
	def __init__(self, features, eps=1e-6):
		super(LayerNorm,self).__init__()
		self.a_2 = nn.Parameter(torch.ones(features))
		self.b_2 = nn.Parameter(torch.zeros(features))
		self.eps = eps

	def forward(self, x):
		mean = x.mean(-1, keepdim =True)
		std = x.std(-1, keepdim=True)   

		return self.a_2 * (x-mean)/ (std + self.eps) + self.b_2


class AddAndNorm(nn.Module):
    def __init__(self, d_model, drop_prob):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x, sublayer_module):
        return x + self.dropout(sublayer_module(self.norm(x)))
  

class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(FeedForward,self).__init__()
        self.w_1 = nn.Linear(d_model, d_model * 4)
        self.w_2 = nn.Linear(d_model * 4, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.w_2(self.dropout(self.relu(self.w_1(x))))
     

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, drop_prob):
        super(MultiHeadAttention, self).__init__()
        self.head_dimension = int(d_model / num_heads)
        self.num_heads = num_heads

        self.qkv_nets = get_clones(nn.Linear(d_model, d_model), 3)
        self.out_projection_net = nn.Linear(d_model, d_model)

        self.attention_dropout = nn.Dropout(p=drop_prob)
        self.softmax = nn.Softmax(dim=-1)
        self.attention_weights = None

    def attention(self, query, key, value, mask):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dimension)

        if mask is not None:
            scores.masked_fill_(mask == torch.tensor(False), float("-inf"))

        attention_weights = self.softmax(scores)
        attention_weights = self.attention_dropout(attention_weights)
        intermediate_token_representations = torch.matmul(attention_weights, value)

        return intermediate_token_representations, attention_weights

    def forward(self, query, key, value, mask):
        batch_size = query.shape[0]
        query, key, value = [net(x).view(batch_size, -1, self.num_heads, self.head_dimension).transpose(1, 2)
                             for net, x in zip(self.qkv_nets, (query, key, value))]

        intermediate_token_representations, attention_weights = self.attention(query, key, value, mask)
        reshaped = intermediate_token_representations.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dimension)
        token_representations = self.out_projection_net(reshaped)

        return token_representations