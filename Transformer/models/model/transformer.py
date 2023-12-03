import torch
from torch import nn

from models.model.encoder import Encoder
from models.model.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_vocab_size, dec_vocab_size, d_model, num_heads, max_seq_len, ff_hidden, num_layers, drop_prob, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device

        self.encoder = Encoder(d_model=d_model,
                               num_heads=num_heads,
                               num_layers=num_layers,
                               max_seq_len=max_seq_len,
                               ff_hidden=ff_hidden,
                               enc_vocab_size=enc_vocab_size,
                               drop_prob=drop_prob,
                               device=device)
        
        self.decoder = Decoder(d_model=d_model,
                               num_heads=num_heads,
                               num_layers=num_layers,
                               max_seq_len=max_seq_len,
                               ff_hidden=ff_hidden,
                               dec_vocab_size=dec_vocab_size,
                               drop_prob=drop_prob,
                               device=device)


    def make_src_mask(self, src):
        ## src : (batch_size, seq_len, d_model)
        ## src_pad_mask : (batch_size, 1, 1, seq_len, d_model) boolean tensor
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2) ## pad_idx가 아닌 원소는 True. pad_idx인 원소는 False인 마스크 생성.

        return src_mask
    
    def make_trg_mask(self, trg):
        ## trg : (batch_size, seq_len, d_model)

        ## pad mask.
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3) ## pad_idx가 아닌 원소는 True. pad_idx인 원소는 False인 마스크 생성.
        trg_len = trg.shape[1] ## TRG Sequence length를 가져와서 mask 차원을 결정.

        ## Look Ahead mask
        ## Lower Triangular matrix를 만들어 현재보다 이전의 위치에만 Attention이 가능하도록하는 mask 생성.
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        
        ## 두 개의 mask를 결합.
        trg_mask = trg_pad_mask & trg_sub_mask

        return trg_mask
    
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src) ## Pad Mask
        trg_mask = self.make_trg_mask(trg) ## Pad Mask + Look Ahead Mask
        
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)

        return output