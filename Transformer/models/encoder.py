from torch import nn
from models.utils import get_clones
from models.sub_layers import AddAndNorm

class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        assert isinstance(encoder_layer, EncoderLayer), f'Expected EncoderLayer got {type(encoder_layer)}.'

        self.encoder_layers = get_clones(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(encoder_layer.d_model)

    def forward(self, src_embeddings_batch, src_mask):
        src_representations_batch = src_embeddings_batch

        for encoder_layer in self.encoder_layers:
            src_representations_batch = encoder_layer(src_representations_batch, src_mask)

        return self.norm(src_representations_batch)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, drop_prob, multi_headed_attention, pointwise_net):
        super().__init__()
        self.d_model = d_model
        num_of_sublayers_encoder = 2
        self.sublayers = get_clones(AddAndNorm(d_model, drop_prob), num_of_sublayers_encoder)

        self.multi_headed_attention = multi_headed_attention
        self.pointwise_net = pointwise_net


    def forward(self, src_representations_batch, src_mask):
        encoder_self_attention = lambda srb: self.multi_headed_attention(query=srb, key=srb, value=srb, mask=src_mask)

        src_representations_batch = self.sublayers[0](src_representations_batch, encoder_self_attention)
        src_representations_batch = self.sublayers[1](src_representations_batch, self.pointwise_net)

        return src_representations_batch