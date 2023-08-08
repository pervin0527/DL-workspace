import copy
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, decoder_block, num_layer, norm):
        super(Decoder, self).__init__()
        self.num_layer = num_layer
        self.layers = nn.ModuleList([copy.deepcopy(decoder_block) for _ in range(self.num_layer)])
        self.norm = norm


    def forward(self, trg, encoder_out, trg_mask, src_trg_mask):
        out = trg
        for layer in self.layers:
            out = layer(out, encoder_out, trg_mask, src_trg_mask)
        out = self.norm(out)

        return out
