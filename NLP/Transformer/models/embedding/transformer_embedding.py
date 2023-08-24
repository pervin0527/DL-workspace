import torch.nn as nn

class TransformerEmbedding(nn.Module):
    def __init__(self, token_embed, pos_embed, drop_prob=0):
        super(TransformerEmbedding, self).__init__()
        self.embedding = nn.Sequential(token_embed, pos_embed)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        out = x
        out = self.embedding(out)
        out = self.dropout(out)

        return out
