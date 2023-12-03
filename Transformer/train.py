import os
import time
import math
import torch

from tqdm import tqdm
from torch import nn
from torchtext.data.utils import get_tokenizer
from nltk.translate.bleu_score import corpus_bleu

from config import *
from data import Multi30kDataset
from models.model.transformer import Transformer
from util.train_util import epoch_time
from util.bleu import idx_to_word, get_bleu


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight.data)


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for src, trg in tqdm(iterator, desc="Training", total=len(iterator)):
        src, trg = src.to(device), trg.to(device)
        # print(src.shape, trg.shape)

        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output_reshape, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        # print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())

    return epoch_loss / len(iterator)


def eval(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    hypotheses = []
    references = []

    with torch.no_grad():
        for batch in tqdm(iterator, desc="Evaluating", total=len(iterator)):
            src, trg = batch[0].to(device), batch[1].to(device)
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()

            output = output.argmax(dim=2)  # 가장 높은 확률의 인덱스 선택

            # BLEU 점수 계산을 위한 가설과 참조 수집
            for j in range(src.size(0)):
                trg_sentence = [data_loader.trg_vocab.get_itos()[idx] for idx in trg[j].unsqueeze(0)]
                output_sentence = [data_loader.trg_vocab.get_itos()[idx] for idx in output[j].tolist()]

                hypotheses.append(output_sentence)
                references.append([trg_sentence])

    # 전체 데이터셋에 대한 BLEU 점수 계산
    bleu_score = corpus_bleu(references, hypotheses) * 100

    return epoch_loss / len(iterator), bleu_score


def run(total_epoch, best_loss):
    if not os.path.isdir(save_dir):
        os.makedirs(f"{save_dir}/result")
        os.makedirs(f"{save_dir}/saved")

    train_loss_list, valid_loss_list, bleu_list = [], [], []
    for epoch in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, train_iter, optimizer, criterion, clip)
        valid_loss, bleu = eval(model, valid_iter, criterion)
        end_time = time.time()

        if epoch > warmup:
            scheduler.step(valid_loss)
        
        train_loss_list.append(train_loss), valid_loss_list.append(valid_loss), bleu_list.append(bleu)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(valid_loss))

        f = open('result/train_loss.txt', 'w')
        f.write(str(train_loss_list))
        f.close()

        f = open('result/bleu.txt', 'w')
        f.write(str(bleu_list))
        f.close()

        f = open('result/test_loss.txt', 'w')
        f.write(str(valid_loss_list))
        f.close()

        print(f'Epoch: {epoch + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')


if __name__ == "__main__":
    tokenize_en = get_tokenizer('spacy', language='en_core_web_sm')
    tokenize_de = get_tokenizer('spacy', language='de_core_news_sm')

    # Create a DataLoaderWrapper instance
    data_loader = Multi30kDataset(ext=('de', 'en'), 
                                  tokenize_en=tokenize_en, 
                                  tokenize_de=tokenize_de, 
                                  sos_token='<sos>', 
                                  eos_token='<eos>', 
                                  batch_size=batch_size,
                                  batch_first=True, 
                                  device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Get iterators
    train_iter, valid_iter = data_loader.make_iter()
    enc_vocab_size, dec_vocab_size = len(data_loader.src_vocab), len(data_loader.trg_vocab)
    src_pad_idx = data_loader.src_vocab.get_stoi()["<pad>"]
    trg_pad_idx = data_loader.trg_vocab.get_stoi()["<pad>"]
    trg_sos_idx = data_loader.trg_vocab.get_stoi()["<sos>"]

    model = Transformer(src_pad_idx=src_pad_idx,
                        trg_pad_idx=trg_pad_idx,
                        trg_sos_idx=trg_sos_idx,
                        d_model=d_model,
                        enc_vocab_size=enc_vocab_size,
                        dec_vocab_size=dec_vocab_size,
                        max_seq_len=max_seq_len,
                        ff_hidden=ff_hidden,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        drop_prob=drop_prob,
                        device=device).to(device)
    
    model.apply(initialize_weights)
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=init_lr,
                                 weight_decay=weight_decay,
                                 eps=adam_eps)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                           verbose=True,
                                                           factor=factor,
                                                           patience=patience)

    criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)

    run(epochs, inf)