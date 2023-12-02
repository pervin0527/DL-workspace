from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward

class DecoderBlock(nn.Module):
    def __init__(self, d_model, ff_hidden, num_heads, drop_prob):
        super(DecoderBlock, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.pwff = PositionwiseFeedForward(d_model=d_model, hidden=ff_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, trg_mask, src_mask):
        ## 1.Decoder Self Attention
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)

        ## 2.Add & Norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        ## 3.Encoder - Decoder Attention
        if enc is not None:
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)

        ## 4.Add & Norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)

        ## 5.position-wise feed forward
        _x = x
        x = self.pwff(x)

        ## 6.Add & Norm
        x = self.dropout3(x)
        x = self.norm3(x + _x)

        return x