import math
import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate, tie_weights, model_type='RNN'):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # RNN, LSTM, GRU 중 하나를 선택하여 사용
        if model_type == 'RNN':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout_rate, batch_first=True)
        elif model_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout_rate, batch_first=True)
        elif model_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout_rate, batch_first=True)
        else:
            raise ValueError("Unsupported RNN type. Choose from 'RNN', 'LSTM', or 'GRU'.")

        """
         임베딩 레이어와 출력 레이어 간의 가중치를 공유하는 기능을 수행합니다. 
            - 임베딩 레이어와 출력 레이어에서 동일한 가중치를 사용함으로써, 모델의 전체 파라미터 수가 감소합니다. 
            - 입력 단어와 출력 단어 간의 임베딩이 동일하게 유지되어, 모델이 더 일관된 표현을 학습하는 데 도움이 될 수 있습니다.
            - 파라미터 수가 감소함으로써 모델이 트레이닝 데이터에 과적합되는 것을 방지하는 데 도움을 줄 수 있습니다.
        """
        if tie_weights:
            assert embedding_dim == hidden_dim, 'If tying weights then embedding_dim must equal hidden_dim'
            self.embedding.weight = self.fc.weight

        self.init_weights()

    def forward(self, src, hidden):
        embedding = self.dropout(self.embedding(src))
        output, hidden = self.rnn(embedding, hidden)
        output = self.dropout(output)
        prediction = self.fc(output)
        return prediction, hidden

    def init_weights(self):
        init_range_emb = 0.1
        init_range_other = 1/math.sqrt(self.hidden_dim)
        self.embedding.weight.data.uniform_(-init_range_emb, init_range_emb)
        self.fc.weight.data.uniform_(-init_range_other, init_range_other)
        self.fc.bias.data.zero_()
        # LSTM과 GRU의 경우 가중치 초기화 방법이 다를 수 있음을 고려할 것
        for i in range(self.num_layers):
            for weight in self.rnn.all_weights[i]:
                weight.data.uniform_(-init_range_other, init_range_other)

    def init_hidden(self, batch_size, device):
        if isinstance(self.rnn, nn.LSTM):
            # LSTM의 경우 hidden state와 cell state 둘 다 초기화
            hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
                      torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device))
        else:
            # RNN과 GRU의 경우 hidden state만 초기화
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return hidden

    def detach_hidden(self, hidden):
        if isinstance(self.rnn, nn.LSTM):
            hidden = (hidden[0].detach(), hidden[1].detach())
        else:
            hidden = hidden.detach()
        return hidden
