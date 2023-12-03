import torch
from torch import nn

from models.blocks.decoder_block import DecoderBlock
from models.embedding.transformer_embedding import TransformerEmbedding

class Decoder(nn.Module):
    def __init__(self, dec_vocab_size, max_seq_len, d_model, ff_hidden, num_heads, num_layers, drop_prob, device):
        super().__init__()
        self.embedd = TransformerEmbedding(d_model=d_model,
                                           max_seq_len=max_seq_len,
                                           vocab_size=dec_vocab_size,
                                           drop_prob=drop_prob,
                                           device=device)
        
        self.layers = nn.ModuleList([DecoderBlock(d_model=d_model, 
                                                  ff_hidden=ff_hidden,
                                                  num_heads=num_heads,
                                                  drop_prob=drop_prob) for _ in range(num_layers)])
        
        self.linear = nn.Linear(d_model, dec_vocab_size)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        """
        enc_src : Context(output of Encoder)
        trg_mask : Look-Ahead Mask + Pad Mask가 적용된 TRG Sequence
        src_mask : Pad Mask가 적용된 SRC Sequence. Context에도 Pad Mask를 적용.
        """
        trg = self.embedd(trg)

        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        output = self.linear(trg)

        return output