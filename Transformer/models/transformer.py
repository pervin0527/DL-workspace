from torch import nn
from models.embedding import Embedding, PositionalEncoding
from models.sub_layers import MultiHeadAttention, FeedForward
from models.encoder import Encoder, EncoderLayer
from models.decoder import Decoder, DecoderLayer

class DecoderGenerator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()

        self.linear = nn.Linear(d_model, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, trg_representations_batch):
        return self.softmax(self.linear(trg_representations_batch))

class Transformer(nn.Module):
    def __init__(self, d_model, src_vocab_size, trg_vocab_size, num_heads, num_layers, max_seq_len=5000, drop_prob=0.1):
        super(Transformer, self).__init__()
        self.src_embedding = Embedding(src_vocab_size, d_model)
        self.trg_embedding = Embedding(trg_vocab_size, d_model)

        self.src_pos_embedding = PositionalEncoding(d_model, max_seq_len=max_seq_len, drop_prob=drop_prob)
        self.trg_pos_embedding = PositionalEncoding(d_model, max_seq_len=max_seq_len, drop_prob=drop_prob)

        mha = MultiHeadAttention(d_model, num_heads, drop_prob)
        pwn = FeedForward(d_model, drop_prob)
        encoder_layer = EncoderLayer(d_model, drop_prob, mha, pwn)
        decoder_layer = DecoderLayer(d_model, drop_prob, mha, pwn)

        self.encoder = Encoder(encoder_layer, num_layers)
        self.decoder = Decoder(decoder_layer, num_layers)
        self.decoder_generator = DecoderGenerator(d_model, trg_vocab_size)

        self.init_params()


    def forward(self, src, trg_input, src_mask, trg_mask):
        src_representations_batch = self.encode(src, src_mask)
        trg_log_probs = self.decode(trg_input, src_representations_batch, trg_mask, src_mask)

        return trg_log_probs


    def encode(self, src_token_ids_batch, src_mask):
        src_embeddings_batch = self.src_embedding(src_token_ids_batch)
        src_embeddings_batch = self.src_pos_embedding(src_embeddings_batch)
        src_representations_batch = self.encoder(src_embeddings_batch, src_mask)

        return src_representations_batch
    

    def decode(self, trg_token_ids_batch, src_representations_batch, trg_mask, src_mask):
        trg_embeddings_batch = self.trg_embedding(trg_token_ids_batch)
        trg_embeddings_batch = self.trg_pos_embedding(trg_embeddings_batch)
        
        trg_representations_batch = self.decoder(trg_embeddings_batch, src_representations_batch, trg_mask, src_mask)
        trg_log_probs = self.decoder_generator(trg_representations_batch)
        
        return trg_log_probs


    def init_params(self, default_initialization=False):
        # Not mentioned in the paper, but other implementations used xavier.
        # I tested both PyTorch's default initialization and this, and xavier has tremendous impact! I didn't expect
        # a model's perf, with normalization layers, to be so much dependent on the choice of weight initialization.
        if not default_initialization:
            for name, p in self.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)