import torch
import copy
import torch.nn as nn

from models.embedding.token_embedding import TokenEmbedding
from models.embedding.positional_encoding import PositionalEncoding
from models.embedding.transformer_embedding import TransformerEmbedding

from models.layer.multi_head_attention_layer import MultiHeadAttentionLayer
from models.layer.position_wise_feed_forward_layer import PositionWiseFeedForwardLayer

from models.model.encoder import Encoder
from models.model.decoder import Decoder
from models.block.encoder_block import EncoderBlock
from models.block.decoder_block import DecoderBlock

from models.model.transformer import Transformer


def build_model(src_vocab_size, trg_vocab_size, max_seq_len=256,
                d_embed=512, d_model=512, num_layer=6, num_heads=8, d_ff=2048, norm_eps=1e-5,
                drop_prob=0.1, device=torch.device("cpu")):

    ## Word Embedding.
    src_token_embed = TokenEmbedding(d_embed=d_embed, vocab_size=src_vocab_size)
    trg_token_embed = TokenEmbedding(d_embed=d_embed, vocab_size=trg_vocab_size)

    ## Positional Encoding.
    src_pos_embedd = PositionalEncoding(d_embed=d_embed, max_seq_len=max_seq_len, device=device)
    trg_pos_embedd = PositionalEncoding(d_embed=d_embed, max_seq_len=max_seq_len, device=device)

    ## Word Embedding + Positional Encoding.
    trg_embed = TransformerEmbedding(token_embed=trg_token_embed, pos_embed=trg_pos_embedd, drop_prob=drop_prob)
    src_embed = TransformerEmbedding(token_embed=src_token_embed, pos_embed=src_pos_embedd, drop_prob=drop_prob)

    ## Multi-Head Self Attention.
    encoder_attention = MultiHeadAttentionLayer(d_model=d_model, d_embed=d_embed, num_heads=num_heads, drop_prob=drop_prob)
    decoder_attention = MultiHeadAttentionLayer(d_model=d_model, d_embed=d_embed, num_heads=num_heads, drop_prob=drop_prob)
    
    ## Position-Wise FeedForward.
    encoder_position_ff = PositionWiseFeedForwardLayer(d_embed, d_ff, drop_prob=drop_prob)
    decoder_position_ff = PositionWiseFeedForwardLayer(d_embed, d_ff, drop_prob=drop_prob)

    ## Add & Norm.
    encoder_norm = nn.LayerNorm(d_embed, eps=norm_eps)
    decoder_norm = nn.LayerNorm(d_embed, eps=norm_eps)

    ## Encoder Block.
    encoder_block = EncoderBlock(self_attention=encoder_attention,
                                 position_ff=encoder_position_ff,
                                 norm=encoder_norm,
                                 drop_prob=drop_prob)
    
    ## Decoder Block
    decoder_block = DecoderBlock(self_attention=decoder_attention,
                                 cross_attention=decoder_attention,
                                 position_ff=decoder_position_ff,
                                 norm=decoder_norm,
                                 drop_prob=drop_prob)

    ## Encoder(Encoder Block * Num_layers)
    encoder = Encoder(encoder_block=encoder_block, num_layer=num_layer, norm=encoder_norm)

    ## Decoder(Decoder Block * Num_layers)
    decoder = Decoder(decoder_block=decoder_block, num_layer=num_layer, norm=decoder_norm)

    ## Output Layer.
    generator = nn.Linear(d_model, trg_vocab_size)

    model = Transformer(src_embed=src_embed,
                        trg_embed=trg_embed,
                        encoder=encoder,
                        decoder=decoder,
                        generator=generator).to(device)
    
    model.device = device

    return model
