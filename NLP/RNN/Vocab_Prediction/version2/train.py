import os
import torch
import random

from torch import nn, optim

from config import *
from model import RNN
from dataset import Multi30k

def train(model, data_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for idx, (_, X, Y) in enumerate(data_loader):
        hidden_state = nn.init.xavier_uniform_(torch.zeros(1, X.size(0), HIDDEN_DIM)).to(DEVICE)

        optimizer.zero_grad()                
        X, Y = X.to(DEVICE), Y.to(DEVICE)
        outputs, _ = model(X, hidden_state)
        loss = criterion(outputs.view(-1, outputs.size(-1)), Y.view(-1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for idx, (sentences, X, Y) in enumerate(data_loader):
            hidden_state = nn.init.xavier_uniform_(torch.zeros(1, X.size(0), HIDDEN_DIM)).to(DEVICE)

            X, Y = X.to(DEVICE), Y.to(DEVICE)
            outputs, _ = model(X, hidden_state)
            loss = criterion(outputs.view(-1, outputs.size(-1)), Y.view(-1))
            total_loss += loss.item()

            rand_idx = random.randint(0, X.size(0) - 1)
            original_sentence = sentences[rand_idx]

            first_vocab = X[rand_idx].argmax(dim=-1).cpu().numpy()
            first_vocab = total_vocabs[first_vocab[0]]

            predicted_sentence = outputs[rand_idx].argmax(dim=-1).cpu().numpy()
            predicted_sentence = f"({first_vocab}) " + ' '.join([total_vocabs[idx] for idx in predicted_sentence])
        
    return total_loss / len(data_loader), original_sentence, predicted_sentence


if __name__ == "__main__":
    DATASET = Multi30k(DATA_DIR, TARGET_LANGUAGE, MAX_SEQ_LEN, MIN_FREQ)
    train_iter, valid_iter, test_iter = DATASET.get_iter(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    total_vocabs = DATASET.vocab.get_itos()

    INPUT_DIM = len(DATASET.vocab)
    OUTPUT_DIM = len(DATASET.vocab)
    HIDDEN_DIM = 512

    model = RNN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=DATASET.PAD_IDX)

    for epoch in range(EPOCHS):
        train_loss = train(model, train_iter, optimizer, criterion)
        valid_loss, original_sentence, predict_sentence = evaluate(model, valid_iter, criterion)
        
        print(f"Epoch : {epoch} | train_loss : {train_loss:.4f}, valid_loss : {valid_loss:.4f}")
        print(f"Original Sentence : {original_sentence}")
        print(f"Predict Sentence : {predict_sentence} \n")