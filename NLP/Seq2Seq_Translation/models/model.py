import random
import torch
import torch.nn as nn
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.lstm = nn.GRU(hidden_dim, hidden_dim)

    def forward(self, input, hidden_state):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden_state = self.lstm(embedded, hidden_state)

        return output, hidden_state

    def init_hidden_state(self, device):
        return torch.zeros(1, 1, self.hidden_dim, device=device)
    

class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.embedding = nn.Embedding(output_dim, hidden_dim)
        self.lstm = nn.GRU(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden_state):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden_state = self.lstm(output, hidden_state)
        output = self.softmax(self.out(output[0]))

        return output, hidden_state
    
    def init_hidden_state(self, device):
        return torch.zeros(1, 1, self.hidden_dim, device=device)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [src_len, batch_size]
        # trg: [trg_len, batch_size]

        trg_len, batch_size = trg.shape
        trg_vocab_size = self.decoder.output_dim

        # 텐서를 저장할 장소를 초기화합니다
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # 인코더의 마지막 hidden state가 디코더의 첫 hidden state가 됩니다
        hidden_state = self.encoder.init_hidden_state(self.device)
        _, hidden_state = self.encoder(src, hidden_state)

        # 디코더의 첫 입력은 <sos> 토큰입니다
        input = trg[0,:]

        for t in range(1, trg_len):
            output, hidden_state = self.decoder(input, hidden_state)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1) 
            input = trg[t] if teacher_force else top1

        return outputs
