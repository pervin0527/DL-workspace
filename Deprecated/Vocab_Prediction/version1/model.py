import torch
from torch import nn 

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=layers, batch_first=True, nonlinearity="relu")
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x):
        x, h = self.rnn(x) ## current_output, current_status
        x = self.fc(x)

        return x

# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(RNN, self).__init__()

#         self.hidden_size = hidden_size
#         self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
#         self.h2o = nn.Linear(hidden_size, output_size)
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, input, hidden):
#         combined = torch.cat((input, hidden), 1)
#         hidden = self.i2h(combined)
#         output = self.h2o(hidden)
#         output = self.softmax(output)

#         return output, hidden

#     def initHidden(self):
#         return torch.zeros(1, self.hidden_size)