import math
import time
import torch
import pandas as pd

from torch import nn
from konlpy.tag import Mecab
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from sklearn.model_selection import train_test_split

from data.utils import get_total_data
from data.dataset import tokenize, build_vocab, tokens_to_indices, TranslationDataset
from models.model import Encoder, Decoder, Seq2Seq


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def evaluate(model, iterator, criterion):
    model.eval()
    
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src, trg = batch[0], batch[1]
            src, trg = src.to(device), trg.to(device)

            output = model(src, trg, 0) #turn off teacher forcing

            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src, trg = batch[0], batch[1]
        src, trg = src.to(device), trg.to(device)
        
        optimizer.zero_grad()
        output = model(src, trg)
        
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "/home/pervinco/Datasets/KORENG"
    epochs = 1000
    batch_size = 32
    learning_rate = 0.01
    clip = 1

    ## Load Data
    df = get_total_data(data_dir)

    ## Data Slcing & split
    df_shuffled=df.sample(frac=1).reset_index(drop=True)
    df = df_shuffled[:10000]
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=False)

    print('train size: ', len(train_df))
    print('valid size: ', len(valid_df))

    ## Tokeninzer
    en_tokenizer = get_tokenizer('basic_english')
    ko_tokenizer = Mecab()

    train_ko_tokens, train_en_tokens = tokenize(train_df, ko_tokenizer, en_tokenizer)
    valid_ko_tokens, valid_en_tokens = tokenize(valid_df, ko_tokenizer, en_tokenizer)

    train_ko_vocab = build_vocab(train_ko_tokens)
    train_en_vocab = build_vocab(train_en_tokens)

    valid_ko_vocab = build_vocab(valid_ko_tokens)
    valid_en_vocab = build_vocab(valid_en_tokens)

    train_ko_indices = [tokens_to_indices(tokens, train_ko_vocab) for tokens in train_ko_tokens]
    train_en_indices = [tokens_to_indices(tokens, train_en_vocab) for tokens in train_en_tokens]

    valid_ko_indices = [tokens_to_indices(tokens, valid_ko_vocab) for tokens in valid_ko_tokens]
    valid_en_indices = [tokens_to_indices(tokens, valid_en_vocab) for tokens in valid_en_tokens]

    train_dataset = TranslationDataset(train_ko_indices, train_en_indices)
    valid_dataset = TranslationDataset(valid_ko_indices, valid_en_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=valid_dataset.collate_fn)

    input_dim = len(train_ko_vocab)
    output_dim = len(train_en_vocab)
    ko_emb_dim = 256
    en_emb_dim = 256
    hidden_dim = 512
    num_layers = 2
    ko_dropout = 0.5
    en_dropout = 0.5

    enc = Encoder(input_dim, ko_emb_dim, hidden_dim, num_layers, ko_dropout)
    dec = Decoder(output_dim, en_emb_dim, hidden_dim, num_layers, en_dropout)

    model = Seq2Seq(enc, dec, device).to(device)
    model.apply(init_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)    
    target_pad_idx = train_en_vocab.get_itos().index("<pad>")
    criterion = nn.CrossEntropyLoss(ignore_index=target_pad_idx)

    best_valid_loss = float('inf')

    for epoch in range(epochs):
        
        start_time = time.time()
        
        train_loss = train(model, train_loader, optimizer, criterion, clip)
        valid_loss = evaluate(model, valid_loader, criterion)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut1-model.pt')
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')