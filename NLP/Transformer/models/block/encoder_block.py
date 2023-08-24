import copy
import torch.nn as nn
from models.layer.residual_connection_layer import ResidualConnectionLayer

class EncoderBlock(nn.Module):
    def __init__(self, self_attention, position_ff, norm, drop_prob=0):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.residual1 = ResidualConnectionLayer(copy.deepcopy(norm), drop_prob)
        self.position_ff = position_ff
        self.residual2 = ResidualConnectionLayer(copy.deepcopy(norm), drop_prob)

    def forward(self, src, src_mask):
        out = src
        out = self.residual1(out, lambda out : self.self_attention(query=out, key=out, value=out, mask=src_mask))
        out = self.residual2(out, self.position_ff)

        return out
