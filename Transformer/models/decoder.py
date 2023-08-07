import copy
from torch import nn
from models.utils import get_clones
from models.sub_layers import AddAndNorm

class Decoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        assert isinstance(decoder_layer, DecoderLayer), f'Expected DecoderLayer got {type(decoder_layer)}.'

        self.decoder_layers = get_clones(decoder_layer, num_layers)
        self.norm = nn.LayerNorm(decoder_layer.d_model)

    def forward(self, trg_embeddings_batch, src_representations_batch, trg_mask, src_mask):
        trg_representations_batch = trg_embeddings_batch

        for decoder_layer in self.decoder_layers:
            trg_representations_batch = decoder_layer(trg_representations_batch, src_representations_batch, trg_mask, src_mask)

        return self.norm(trg_representations_batch)
    

class DecoderLayer(nn.Module):
    def __init__(self, d_model, drop_prob, multi_headed_attention, pointwise_net):
        super().__init__()
        self.d_model = d_model
        num_of_sublayers_decoder = 3
        self.sublayers = get_clones(AddAndNorm(d_model, drop_prob), num_of_sublayers_decoder)

        self.trg_multi_headed_attention = copy.deepcopy(multi_headed_attention)
        self.src_multi_headed_attention = copy.deepcopy(multi_headed_attention)
        self.pointwise_net = pointwise_net

    def forward(self, trg_representations_batch, src_representations_batch, trg_mask, src_mask):
        srb = src_representations_batch
        decoder_trg_self_attention = lambda trb: self.trg_multi_headed_attention(query=trb, key=trb, value=trb, mask=trg_mask)
        decoder_src_attention = lambda trb: self.src_multi_headed_attention(query=trb, key=srb, value=srb, mask=src_mask)

        trg_representations_batch = self.sublayers[0](trg_representations_batch, decoder_trg_self_attention)
        trg_representations_batch = self.sublayers[1](trg_representations_batch, decoder_src_attention)
        trg_representations_batch = self.sublayers[2](trg_representations_batch, self.pointwise_net)

        return trg_representations_batch