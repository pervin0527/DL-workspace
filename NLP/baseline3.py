import torch
import numpy as np

# declare RNN + FC
class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layers):
        super(Net, self).__init__()
        self.rnn = torch.nn.RNN(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x):
        x, _status = self.rnn(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    sentence = ("if you want to build a ship, don't drum up people together to "
                "collect wood and don't assign them tasks and work, but rather "
                "teach them to long for the endless immensity of the sea.")

    char_set = list(set(sentence)) ## unique character list
    char_dict = {c: i for i, c in enumerate(char_set)}

    dict_size = len(char_dict)
    hidden_size = len(char_dict)
    sequence_length = 10  # Any arbitrary number
    learning_rate = 0.1

    # data setting
    x_data = []
    y_data = []

    for i in range(0, len(sentence) - sequence_length):
        x_str = sentence[i:i + sequence_length]
        y_str = sentence[i + 1: i + sequence_length + 1]
        # print(i, x_str, '->', y_str)

        x_data.append([char_dict[c] for c in x_str])  # x str to index
        y_data.append([char_dict[c] for c in y_str])  # y str to index

    x_one_hot = np.array([np.eye(dict_size)[x] for x in x_data])

    # transform as torch tensor variable
    X = torch.FloatTensor(x_one_hot)
    Y = torch.LongTensor(y_data)

    net = Net(dict_size, hidden_size, 2)

    # loss & optimizer setting
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), learning_rate)

    # start training
    for i in range(500):
        optimizer.zero_grad()
        outputs = net(X)
        loss = criterion(outputs.view(-1, dict_size), Y.view(-1))
        loss.backward()
        optimizer.step()

        results = outputs.argmax(dim=2)
        predict_str = ""
        for j, result in enumerate(results):
            # print(i, j, ''.join([char_set[t] for t in result]), loss.item())
            if j == 0:
                predict_str += ''.join([char_set[t] for t in result])
            else:
                predict_str += char_set[result[-1]]

        print(predict_str)