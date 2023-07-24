import os
import glob
import time
import math
import torch
import random
import string
import unicodedata
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from io import open


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor


# 유니코드 문자열을 ASCII로 변환, https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn' and c in all_letters)


def findFiles(path): 
    return glob.glob(path)


def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn' and c in all_letters)


# 파일을 읽고 줄 단위로 분리
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


# all_letters 로 문자의 주소 찾기, 예시 "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)


# 검증을 위해서 한 개의 문자를 <1 x n_letters> Tensor로 변환
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


# 한 줄(이름)을  <line_length x 1 x n_letters>,
# 또는 One-Hot 문자 벡터의 Array로 변경
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


def categoryFromOutput(output):
    top_n, top_i = output.topk(1)  # 텐서의 가장 큰 값 및 주소
    category_i = top_i[0].item()   # 텐서에서 정수 값으로 변경
    return all_categories[category_i], category_i


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size): ## 57, 128, 18
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size) ## 185, 128
        self.i2o = nn.Linear(input_size + hidden_size, output_size) ## 185, 18
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


def train(category_tensor, line_tensor):
    hidden = rnn.initHidden() ## 모든 시점마다 hidden state가 0이 되는 것처럼 보이지만 실제로 출력해보면 갱신되고 있음.
    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden) ## current state를 계산할 때 이전 state를 사용한다. 즉 이 시점에 hidden state가 update된다.

    loss = criterion(output, category_tensor)
    loss.backward()

    # weight의 gradient(update 방향 및 크기)에 학습률을 곱해서 그 매개변수의 값에 더함. 즉, parameter update
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()


if __name__ == "__main__":
    dir = "/home/pervinco/Datasets/rnn-tutorial/data/names"
    n_hidden = 128
    learning_rate = 0.005

    all_letters = string.ascii_letters + " .,;'" ## abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,;'
    n_letters = len(all_letters)
    print(f"num of letters {n_letters}")

    category_lines = {} ## 각 언어별 first name
    all_categories = [] ## 언어 list
    for filename in findFiles(f'{dir}/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)

        lines = readLines(filename)
        category_lines[category] = lines

    n_categories = len(all_categories)
    print(f"num of categories {n_categories}")

    rnn = RNN(n_letters, n_hidden, n_categories) ## 57, 128, 18

    input = lineToTensor('Albert') ## 각 문자별 one-hot encoded vector들이 tensor를 형성. input_len * 1 * n_letters
    print(input.shape)
    hidden = torch.zeros(1, n_hidden) ## 1, 128
    print(hidden.shape)
    criterion = nn.NLLLoss()

    n_iters = 100000
    print_every = 5000
    plot_every = 1000
    current_loss = 0
    all_losses = []
    start = time.time()
    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        # print(f"Category, line : {category_tensor.shape}, {line_tensor.shape}") ## category_tensor : [1], line_tensor : [num_chr, 1, n_letters]
        output, loss = train(category_tensor, line_tensor)
        current_loss += loss

        # ``iter`` 숫자, 손실, 이름, 추측 화면 출력
        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

        # 현재 평균 손실을 전체 손실 리스트에 추가
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    plt.figure()
    plt.plot(all_losses)