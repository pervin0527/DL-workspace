import torch
from torch import nn 

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(hidden_dim, output_dim)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x, h0):
        out, hn = self.rnn(x, h0)
        out = self.fc(out)
        # out = self.softmax(out)

        return out, hn