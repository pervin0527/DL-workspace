from torch import nn
from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward

class EncoderBlock(nn.Module):
    def __init__(self, d_model, ff_hidden, num_heads, drop_prob):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.pwff = PositionwiseFeedForward(d_model=d_model, hidden=ff_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_pad_mask):
        ## 1. Self-Attention
        _x = x ## for Residual Connection
        x = self.attention(q=x, k=x, v=x, mask=src_pad_mask)

        ## 2. Add & Norm
        x = self.dropout1(x)
        x = self.norm1(x + _x) ## Residual Connection & LayerNorm

        ## 3. positionwise feed forward
        _x = x ## for Residual Connection
        x = self.pwff(x)

        ## 4. Add & Norm
        x = self.dropout2(x)
        x = self.norm2(x + _x) ## Residual Connection & LayerNorm

        return x