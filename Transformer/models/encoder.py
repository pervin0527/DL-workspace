from torch import nn
from models.layers.encoder_layer import EncoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding

class Encoder(nn.Module):
    def __init__(self, encoder_vocab_size, max_len, d_model, ffn_hidden, num_head, num_layers, drop_prob, device):
        super().__init__()
        self.embedding = TransformerEmbedding(d_model=d_model,
                                              max_len=max_len,
                                              vocab_size=encoder_vocab_size,
                                              drop_prob=drop_prob,
                                              device=device)

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model, ffn_hidden=ffn_hidden, num_head=num_head, drop_prob=drop_prob) for _ in range(num_layers)])

    def forward(self, x, src_mask):
        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x