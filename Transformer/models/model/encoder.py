from torch import nn
from models.embedding.transformer_embedding import TransformerEmbedding

class Encoder(nn.Module):
    def __init__(self, encoder_vocab_size, max_length, d_model, num_heads, ffn_hidden, num_layers, drop_prob, device):
        super().__init__()
        self.embedding = TransformerEmbedding(d_model=d_model,
                                              max_length=max_length,
                                              vocab_size=encoder_vocab_size,
                                              drop_prob=drop_prob,
                                              device=device)