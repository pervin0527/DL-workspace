import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import math
import torch

from tqdm import tqdm
from torch import nn

from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer

from data.utils import get_total_data, split_data
from models.model import init_weights, Encoder, Decoder, Seq2Seq, EncoderGRU, DecoderGRU, Seq2SeqGRU, EncoderBidGRU, AttentionDecoder, Attention, AttentionSeq2Seq
from data.dataset import TranslationDataset, collate_fn, build_vocab, text_transform


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
    TRAIN_RATIO = 0.8

    EPOCHS = 100
    BATCH_SIZE = 64
    LEARNING_RATE = 0.01
    CLIP = 1

    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    print("Load Dataset")
    dataset = get_total_data(DATA_DIR, reverse=False) ## Default : ko -> en
    kor_sentences, eng_sentences = dataset[0], dataset[1]
    kor_sentences, eng_sentences = kor_sentences[:10000], eng_sentences[:10000]
    print(f"Total Sentences | SRC : {len(kor_sentences)}, TRG : {len(eng_sentences)}\n")

    print("Building Vocabs...")
    kor_tokenizer = get_tokenizer('spacy', language='ko_core_news_sm')
    eng_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    kor_vocabs = build_vocab(kor_sentences, kor_tokenizer)
    eng_vocabs = build_vocab(eng_sentences, eng_tokenizer)
    print(f"Vocabs | SRC : {len(kor_vocabs)}, TRG : {len(eng_vocabs)}\n")

    print("Convert Tokenized List to Indices List...")
    kor_indices = [text_transform(sentence, kor_tokenizer, kor_vocabs) for sentence in kor_sentences]
    eng_indices = [text_transform(sentence, eng_tokenizer, eng_vocabs) for sentence in eng_sentences]
    print(f"Indices | SRC : {len(kor_indices)}, TRG : {len(eng_indices)}\n")

    ## Split Dataset
    print("Data Split -> Train / Valid / TEST")
    train_data, valid_data, test_data = split_data(kor_indices, eng_indices, train_frac=0.8, valid_frac=0.1)
    print(f"Train Dataset | SRC : {len(train_data[0])}, TRG : {len(train_data[1])}")
    print(f"Valid Dataset | SRC : {len(valid_data[0])}, TRG : {len(valid_data[1])}")
    print(f"Test Dataset | SRC : {len(test_data[0])}, TRG : {len(test_data[1])}\n")

    ## Define Torch Dataset & DataLoader
    train_dataset = TranslationDataset(train_data)
    valid_dataset = TranslationDataset(valid_data)
    test_dataset = TranslationDataset(test_data)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    ## Encoder, Decoder Params
    # INPUT_DIM = len(kor_vocabs)
    # OUTPUT_DIM = len(eng_vocabs)
    # EMBEDD_DIM = 256
    # HIDDEN_DIM = 512
    # NUM_LAYERS = 3
    # ENCODER_DROPOUT = 0.2
    # DECODER_DROPOUT = 0.2

    ## Define Model
    # encoder = Encoder(input_dim=INPUT_DIM, embedd_dim=EMBEDD_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=ENCODER_DROPOUT)
    # decoder = Decoder(output_dim=OUTPUT_DIM, embedd_dim=EMBEDD_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DECODER_DROPOUT)
    # model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    # encoder = EncoderGRU(INPUT_DIM, EMBEDD_DIM, HIDDEN_DIM, ENCODER_DROPOUT)
    # decoder = DecoderGRU(OUTPUT_DIM, EMBEDD_DIM, HIDDEN_DIM, DECODER_DROPOUT)
    # model = Seq2SeqGRU(encoder, decoder, DEVICE).to(DEVICE)

    INPUT_DIM = len(kor_vocabs)
    OUTPUT_DIM = len(eng_vocabs)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    ENC_HID_DIM = 512
    DEC_HID_DIM = 512
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = EncoderBidGRU(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    dec = AttentionDecoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
    model = AttentionSeq2Seq(enc, dec, DEVICE).to(DEVICE)
    model.apply(init_weights)

    ## Optimizer & Loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    trg_pad_idx = eng_vocabs.get_itos().index("<pad>")
    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

    ## Train Loop
    best_valid_loss = float('inf')
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        train_loss = train(model, train_dataloader, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_dataloader, criterion)
        
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'{SAVE_DIR}/best-koen.pt')
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tValid Loss: {valid_loss:.3f} | Valid PPL: {math.exp(valid_loss):7.3f}\n')