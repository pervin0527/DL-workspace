import os
import torch
import random

from torch import nn, optim

from config import *
from model import RNN
from dataset import Multi30k, preprocessing


def random_choice(dataset):
    idx = random.randint(0, len(dataset) - 1)
    sentence = dataset[idx]
    # sentence = dataset[10]

    x, y = preprocessing(sentence, str_vocabs)

    return sentence, x, y


def compare(outputs):
    results = outputs.argmax(dim=2)
    results_str = ""
    for j, result in enumerate(results):
        if j == 0:
            results_str += ''.join([str_vocabs[x] for x in result])
        else:
            results_str += str_vocabs[result[-1]]

    return results_str


def evaluate():
    model.eval()
    with torch.no_grad():
        sentence, x, y = random_choice(valid_dataset)
        outputs = model(x)

        loss = criterion(outputs.view(-1, INPUT_DIM), y.view(-1))
        predict_sentence = compare(outputs)

    return loss.item(), sentence, predict_sentence


def train():
    model.train()
    current_loss = 0
    for iter in range(1, MAX_ITER + 1):
        sentence, x, y = random_choice(train_dataset)

        optimizer.zero_grad()
        outputs = model(x)
        # print(x.shape, y.shape, outputs.shape)

        loss = criterion(outputs.view(-1, INPUT_DIM), y.view(-1))
        loss.backward()
        optimizer.step()

        current_loss += loss.item()

        if iter % 100 == 0:
            losses = current_loss / 100

            print("==========")
            print(f"iter{iter} | Train_loss : {losses:.4f}")
            predict_sentence = compare(outputs)
            print(f"Original Sentence : {sentence}")
            print(f"Predict Sentence : {predict_sentence} \n")

            valid_loss, origin_sentence, valid_sentence = evaluate()
            print(f"Valid_loss : {valid_loss:.4f}")
            print(f"Original Sentence : {origin_sentence}")
            print(f"Predict Sentence : {valid_sentence} \n")
            current_loss = 0


if __name__ == "__main__":
    DATASET = Multi30k(DATA_DIR, TARGET_LANGUAGE, MAX_SEQ_LEN, MIN_FREQ)
    train_dataset, valid_dataset, _, vocabs = DATASET.get_dataset()
    str_vocabs = list(vocabs.keys())

    INPUT_DIM = len(vocabs)
    HIDDEN_DIM = len(vocabs)
    model = RNN(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train()