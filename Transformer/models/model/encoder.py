from torch import nn
from models.blocks.encoder_block import EncoderBlock
from models.embedding.transformer_embedding import TransformerEmbedding

class Encoder(nn.Module):
    def __init__(self, enc_vocab_size, max_seq_len, d_model, ff_hidden, num_heads, num_layers, drop_prob, device):
        super().__init__()
        self.embedd = TransformerEmbedding(d_model=d_model, 
                                           max_seq_len=max_seq_len,
                                           vocab_size=enc_vocab_size,
                                           drop_prob=drop_prob,
                                           device=device)
        
        self.encider_layers = nn.ModuleList([EncoderBlock(d_model=d_model,
                                                          ff_hidden=ff_hidden, 
                                                          num_heads=num_heads, 
                                                          drop_prob=drop_prob) for _ in range(num_layers)])

    def forward(self, x, src_pad_mask):
        x = self.embedd(x)

        for layer in self.encider_layers:
            x = layer(x, src_pad_mask)

        return x