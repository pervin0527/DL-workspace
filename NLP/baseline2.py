""" charseq """

import torch
import numpy as np

sample = "hihello"
char_set = sorted(list(set(sample)))
char_dict = {c : i for i, c in enumerate(char_set)}

input_size = len(char_dict)
hidden_size = len(char_dict)
learning_rate = 0.1

sample_idx = [char_dict[c] for c in sample]
x_data = [sample_idx[:-1]] ## 마지막 문자는 input으로 사용하지 않고 model이 prediction해야 함.
x_one_hot = [np.eye(input_size)[x] for x in x_data] ## [x] indexing을 통해 해당 row만 가져오도록 한다.
y_data = [sample_idx[1:]] ## 첫번째 문자는 ground-truth에 포함시키지 않는다.

X = torch.FloatTensor(np.array(x_one_hot))
Y = torch.LongTensor(y_data)

rnn = torch.nn.RNN(input_size, hidden_size, batch_first=True) ## batch_fist는 batch dimension을 가장 앞으로 위치하게 한다.

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), learning_rate)

for i in range(100):
    optimizer.zero_grad() ## epoch마다 새로운 gradient를 계산한다. 적용하지 않으면 지속적으로 누적된다.
    outputs, _status = rnn(X) ## 주어진 모든 input을 처리하고 구한 status이기 때문에 current status가 사용되지 않고 있음.
    loss = criterion(outputs.view(-1, input_size), Y.view(-1)) ## dimension matching
    loss.backward()

    optimizer.step() ## weight update
    result = outputs.data.numpy().argmax(axis=2)
    result_str = ''.join([char_set[c] for c in np.squeeze(result)])
    print(i, "loss: ", loss.item(), "prediction: ", result, "true Y: ", y_data, "prediction str: ", result_str, "\n")