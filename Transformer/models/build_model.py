import torch
import copy
import torch.nn as nn

from models.model.transformer import Transformer
from models.model.encoder import Encoder
from models.model.decoder import Decoder
from models.block.encoder_block import EncoderBlock
from models.block.decoder_block import DecoderBlock
from models.layer.multi_head_attention_layer import MultiHeadAttentionLayer
from models.layer.position_wise_feed_forward_layer import PositionWiseFeedForwardLayer
from models.embedding.transformer_embedding import TransformerEmbedding
from models.embedding.token_embedding import TokenEmbedding
from models.embedding.positional_encoding import PositionalEncoding


def build_model(src_vocab_size,
                trg_vocab_size,
                device=torch.device("cpu"),
                max_seq_len=256,
                d_embed=512,
                d_model=512,
                num_layer=6,
                num_heads=8,
                d_ff=2048,
                drop_prob=0.1,
                norm_eps=1e-5):

    src_token_embed = TokenEmbedding(d_embed=d_embed, vocab_size=src_vocab_size)
    trg_token_embed = TokenEmbedding(d_embed=d_embed, vocab_size=trg_vocab_size)
    pos_embed = PositionalEncoding(d_embed=d_embed, max_seq_len=max_seq_len, device=device)

    src_embed = TransformerEmbedding(token_embed=src_token_embed,
                                     pos_embed=copy.deepcopy(pos_embed),
                                     drop_prob=drop_prob)
    
    trg_embed = TransformerEmbedding(token_embed=trg_token_embed,
                                     pos_embed=copy.deepcopy(pos_embed),
                                     drop_prob=drop_prob)

    attention = MultiHeadAttentionLayer(d_model=d_model,
                                        d_embed=d_embed,
                                        num_heads=num_heads,
                                        drop_prob=drop_prob)
    
    position_ff = PositionWiseFeedForwardLayer(d_embed, d_ff, drop_prob=drop_prob)
    norm = nn.LayerNorm(d_embed, eps=norm_eps)

    encoder_block = EncoderBlock(self_attention=copy.deepcopy(attention),
                                 position_ff=copy.deepcopy(position_ff),
                                 norm=copy.deepcopy(norm),
                                 drop_prob=drop_prob)
    
    decoder_block = DecoderBlock(self_attention=copy.deepcopy(attention),
                                 cross_attention=copy.deepcopy(attention),
                                 position_ff=copy.deepcopy(position_ff),
                                 norm=copy.deepcopy(norm),
                                 drop_prob=drop_prob)

    encoder = Encoder(encoder_block=encoder_block, num_layer=num_layer, norm=copy.deepcopy(norm))
    decoder = Decoder(decoder_block=decoder_block, num_layer=num_layer, norm=copy.deepcopy(norm))
    generator = nn.Linear(d_model, trg_vocab_size)

    model = Transformer(src_embed=src_embed,
                        trg_embed=trg_embed,
                        encoder=encoder,
                        decoder=decoder,
                        generator=generator).to(device)
    
    model.device = device

    return model
