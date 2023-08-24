import torch
from torch import nn

def get_attn_pad_mask(seq_q, seq_k, i_pad):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(i_pad).unsqueeze(1).expand(batch_size, len_q, len_k)  # <pad>

    return pad_attn_mask

class ScaledDotProductAttention(nn.Moudle):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        self.scale = 1 / (self.config.head_dim ** 0.5)

    def forward(self, Q, K, V, attention_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)).mul_(self.scale)
        scores.masked_fill_(attention_mask, -1e9)
        attention_prob = nn.Softmax(dim=-1)(scores)
        attention_prob = self.dropout(attention_prob)
        context = torch.matmul(attention_prob, V)

        return context, attention_prob
    
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.W_Q = nn.Linear(self.config.hidden_dim, self.config.num_head * self.config.head_dim)
        self.W_K = nn.Linear(self.config.hidden_dim, self.config.num_head * self.config.head_dim)
        self.W_V = nn.Linear(self.config.hidden_dim, self.config.num_head * self.config.head_dim)
        self.scaled_dot_attn = ScaledDotProductAttention(self.config)
        self.linear = nn.Linear(self.config.num_head * self.config.head_dim, self.config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, Q, K, V, attention_mask):
        batch_size = Q.size(0)
        # (bs, num_head, n_q_seq, head_dim)
        q_s = self.W_Q(Q).view(batch_size, -1, self.config.num_head, self.config.head_dim).transpose(1,2)
        # (bs, num_head, n_k_seq, head_dim)
        k_s = self.W_K(K).view(batch_size, -1, self.config.num_head, self.config.head_dim).transpose(1,2)
        # (bs, num_head, n_v_seq, head_dim)
        v_s = self.W_V(V).view(batch_size, -1, self.config.num_head, self.config.head_dim).transpose(1,2)

        # (bs, num_head, n_q_seq, n_k_seq)
        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.config.num_head, 1, 1)

        # (bs, num_head, n_q_seq, head_dim), (bs, num_head, n_q_seq, n_k_seq)
        context, attn_prob = self.scaled_dot_attn(q_s, k_s, v_s, attention_mask)
        # (bs, num_head, n_q_seq, h_head * head_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.config.num_head * self.config.head_dim)
        # (bs, num_head, n_q_seq, e_embd)
        output = self.linear(context)
        output = self.dropout(output)
        # (bs, n_q_seq, d_hidn), (bs, num_head, n_q_seq, n_k_seq)

        return output, attn_prob