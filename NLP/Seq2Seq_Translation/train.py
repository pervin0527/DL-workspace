import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import math
import torch
import spacy


from torch import nn
from tqdm import tqdm
from konlpy.tag import Mecab
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator

from data.utils import get_total_data, split_data
from data.dataset import TranslationDataset
from models.model import Seq2Seq, Encoder, Decoder

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def tokenize_ko(text):
    return [tok for tok in ko_tokenizer.morphs(text)]


def tokenize_en(text):
    return [tok.text for tok in en_tokenizer.tokenizer(text)]


def build_vocab(data_iter, tokenizer):
    vocab = build_vocab_from_iterator(map(tokenizer, data_iter), specials=["<pad>", "<sos>", "<eos>", "<unk>"], min_freq=2)
    vocab.set_default_index(vocab['<unk>'])

    return vocab


def tokens_to_indices(tokens, vocab):
    return [vocab[token] for token in tokens]


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    
    return elapsed_mins, elapsed_secs


def evaluate(model, iterator, criterion):
    model.eval()
    
    epoch_loss = 0
    with torch.no_grad():
        for batch in tqdm(iterator, desc='Evaluating', leave=False):
            src, trg = batch[0], batch[1]
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            output = model(src, trg, 0)
            
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    
    epoch_loss = 0
    for batch in tqdm(iterator, desc='Training', leave=False):
        src, trg = batch[0], batch[1]
        src, trg = src.to(DEVICE), trg.to(DEVICE)
        
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
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_DIR = "/home/pervinco/Datasets/KORENG"
    SAVE_DIR = "/home/pervinco/Models/KORENG"
    
    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    CLIP = 1

    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    ## Define Tokenizer
    ko_tokenizer = Mecab()
    en_tokenizer = spacy.load('en_core_web_sm')

    ## Load total dataset
    print("Load Dataset")
    dataset = get_total_data(DATA_DIR, reverse=False) ## Default : ko -> en

    src_sentences, trg_sentences = dataset[0][:10000], dataset[1][:10000]
    print(f"Total Sentences | SRC : {len(src_sentences)}, TRG : {len(trg_sentences)}\n")

    ## Tokeinze & Build Vocabs
    print("Building Vocabs...")
    src_vocabs = build_vocab(src_sentences, tokenize_ko)
    trg_vocabs = build_vocab(trg_sentences, tokenize_en)
    print(f"Vocabs | SRC : {len(src_vocabs)}, TRG : {len(trg_vocabs)}\n")

    print("Convert Tokenized List to Indices List...")
    src_indices = [tokens_to_indices(tokens, src_vocabs) for tokens in src_sentences]
    trg_indices = [tokens_to_indices(tokens, trg_vocabs) for tokens in trg_sentences]
    print("Done.\n")

    ## Split Dataset
    print("Data Split -> Train / Valid / TEST")
    train_data, valid_data, test_data = split_data(src_indices, trg_indices, train_frac=0.8, valid_frac=0.1)
    print(f"Train Dataset | SRC : {len(train_data[0])}, TRG : {len(train_data[1])}")
    print(f"Valid Dataset | SRC : {len(valid_data[0])}, TRG : {len(valid_data[1])}")
    print(f"Test Dataset | SRC : {len(test_data[0])}, TRG : {len(test_data[1])}\n")


    ## Define Torch Dataset & DataLoader
    train_dataset = TranslationDataset(train_data)
    valid_dataset = TranslationDataset(valid_data)
    test_dataset = TranslationDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_dataset.collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=valid_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=test_dataset.collate_fn)

    ## Encoder, Decoder Params
    INPUT_DIM = len(src_vocabs)
    OUTPUT_DIM = len(trg_vocabs)
    EMBEDD_DIM = 1024
    HIDDEN_DIM = 2048
    NUM_LAYERS = 4
    ENCODER_DROPOUT = 0.5
    DECODER_DROPOUT = 0.5

    ## Define Model
    encoder = Encoder(input_dim=INPUT_DIM, embedd_dim=EMBEDD_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=ENCODER_DROPOUT)
    decoder = Decoder(output_dim=OUTPUT_DIM, embedd_dim=EMBEDD_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DECODER_DROPOUT)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
    model.apply(init_weights)

    ## Optimizer & Loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    trg_pad_idx = trg_vocabs.get_itos().index("<pad>")
    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

    ## Train Loop
    best_valid_loss = float('inf')

    for epoch in range(EPOCHS):
        
        start_time = time.time()
        
        train_loss = train(model, train_loader, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_loader, criterion)
        
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'{SAVE_DIR}/best-koen.pt')
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tValid Loss: {valid_loss:.3f} | Valid PPL: {math.exp(valid_loss):7.3f}')

    model.load_state_dict(torch.load(f'{SAVE_DIR}/best-koen.pt'))
    test_loss = evaluate(model, test_loader, criterion)
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')