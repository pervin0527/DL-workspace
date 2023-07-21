""" charseq """

import torch
import numpy as np
import torch.nn as nn

class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # Input to hidden layer weights
        self.input_to_hidden = nn.Linear(input_size, hidden_size)
        
        # Hidden to hidden layer weights
        self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size)
        
        # Hidden to output layer weights
        self.hidden_to_output = nn.Linear(hidden_size, output_size)
        
        # Activation function for the hidden layer (tanh or ReLU can also be used)
        self.activation = nn.Tanh()

    def forward(self, input, state):
        # Compute hidden state using the input and previous hidden state
        hidden = self.activation(self.input_to_hidden(input) + self.hidden_to_hidden(state))
        
        # Compute output from the hidden state
        output = self.hidden_to_output(hidden)
        
        return output, hidden
    
    def init_weights(self):
        # Xavier initialization for linear layers
        nn.init.xavier_uniform_(self.input_to_hidden.weight)
        nn.init.xavier_uniform_(self.hidden_to_hidden.weight)
        nn.init.xavier_uniform_(self.hidden_to_output.weight)

        # Initialize biases to zeros
        nn.init.zeros_(self.input_to_hidden.bias)
        nn.init.zeros_(self.hidden_to_hidden.bias)
        nn.init.zeros_(self.hidden_to_output.bias)

if __name__ == "__main__":
    print("About The Data.")
    sample = "hi,hello"
    char_set = sorted(list(set(sample)))
    print(char_set)
    char_dict = {c : i for i, c in enumerate(char_set)}
    print(char_dict)

    input_size = len(char_dict)
    hidden_size = len(char_dict)
    learning_rate = 0.1

    sample_idx = [char_dict[c] for c in sample] ## index list
    print(sample_idx)
    x_data = [sample_idx[:-1]] ## 마지막 문자는 input으로 사용하지 않고 model이 prediction해야 함.
    x_one_hot = [np.eye(input_size)[x] for x in x_data] ## [x] indexing을 통해 해당 row만 가져오도록 한다.
    y_data = [sample_idx[1:]] ## 첫번째 문자는 ground-truth에 포함시키지 않는다.

    X = torch.FloatTensor(np.array(x_one_hot))
    Y = torch.LongTensor(y_data)
    initial_state = torch.zeros(1, 1, hidden_size)

    rnn = torch.nn.RNN(input_size, hidden_size, batch_first=True) ## batch_fist는 batch dimension을 가장 앞으로 위치하게 한다.
    ## Get the weight shapes for each layer
    # rnn = VanillaRNN(input_size=input_size, hidden_size=hidden_size, output_size=hidden_size)
    # rnn.init_weights()
    # input_to_hidden_weights = rnn.input_to_hidden.weight.shape
    # hidden_to_hidden_weights = rnn.hidden_to_hidden.weight.shape
    # hidden_to_output_weights = rnn.hidden_to_output.weight.shape
    # print("\n Layer's weight shape")
    # print(input_to_hidden_weights)
    # print(hidden_to_hidden_weights)
    # print(hidden_to_output_weights, "\n")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), learning_rate)

    print("\n Start Trinaing...")
    for i in range(1000):
        optimizer.zero_grad() ## epoch마다 새로운 gradient를 계산한다. 적용하지 않으면 지속적으로 누적된다.
        # print(X.shape, Y.shape) ## [1, 7, 6], [1, 7]
        outputs, _status = rnn(X, initial_state) ## 주어진 모든 input을 처리하고 구한 status이기 때문에 current status가 사용되지 않고 있음.
        # print(outputs.shape, _status.shape) ## [1, 7, 6] [1, 1, 6]
        
        loss = criterion(outputs.view(-1, input_size), Y.view(-1)) ## dimension matching
        loss.backward()

        optimizer.step() ## weight update
        result = outputs.data.numpy().argmax(axis=2)
        result_str = ''.join([char_set[c] for c in np.squeeze(result)])
        print(i, "loss: ", loss.item(), "prediction: ", result, "true Y: ", y_data, "prediction str: ", result_str, "\n")