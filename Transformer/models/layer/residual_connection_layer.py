import torch.nn as nn

class ResidualConnectionLayer(nn.Module):
    def __init__(self, norm, drop_prob=0):
        super(ResidualConnectionLayer, self).__init__()
        self.norm = norm
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x, sub_layer):
        out = x
        out = self.norm(out)
        out = sub_layer(out)
        out = self.dropout(out)
        out = out + x

        # out = x
        # out = sub_layer(out)
        # out = self.dropout(out)
        # out = self.norm(x + out)
        
        return out
