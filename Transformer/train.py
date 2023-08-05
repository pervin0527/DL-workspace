import os
import math
import torch
import numpy as np
from torch import nn, optim
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k
from collections import Counter

from config import *
from data.dataset import DataGenerator
from models.transformer import Transformer

def bleu_stats(hypothesis, reference):
    """Compute statistics for BLEU."""
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    for n in range(1, 5):
        s_ngrams = Counter([tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)])
        r_ngrams = Counter([tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)])

        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis) + 1 - n, 0]))

    return stats

def bleu(stats):
    """Compute BLEU given n-gram statistics."""
    if len(list(filter(lambda x: x == 0, stats))) > 0:
        return 0
    (c, r) = stats[:2]
    log_bleu_prec = sum(
        [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
    ) / 4.
    return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)


def get_bleu(hypotheses, reference):
    """Get validation BLEU score for dev set."""
    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    for hyp, ref in zip(hypotheses, reference):
        stats += np.array(bleu_stats(hyp, ref))
    return 100 * bleu(stats)

def idx_to_word(x, vocab):
    words = []
    for i in x:
        word = vocab[i]
        if '<' not in word:
            words.append(word)
    words = " ".join(words)
    return words

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight.data)


def train(model, dataloader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    num_iter = len(list(dataloader))
    for i, batch in enumerate(dataloader):
        src, trg = batch[0].transpose(1, 0), batch[1].transpose(1, 0)
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output_reshape, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        # print('step :', round((i / num_iter) * 100, 2), '% , loss :', loss.item())

    return epoch_loss / num_iter


def evaluate(model, dataloader, target_vocab, criterion):
    model.eval()
    epoch_loss = 0
    valid_iter = len(list(dataloader))
    batch_bleu = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            src, trg = batch[0].transpose(1, 0), batch[1].transpose(1, 0)
            src, trg = src.to(device), trg.to(device)

            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()

            total_bleu = []
            for j in range(batch_size):
                try:
                    trg_words = idx_to_word(batch[1].transpose(1, 0)[j], target_vocab)
                    output_words = output[j].max(dim=1)[1]
                    output_words = idx_to_word(output_words, target_vocab)
                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                    total_bleu.append(bleu)
                except:
                    pass

            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    return epoch_loss / valid_iter, batch_bleu


def run(total_epoch, best_loss):
    src_lang = 'en'
    trg_lang = 'de'    
    train_generator = DataGenerator("train", src_lang, trg_lang)
    valid_generator = DataGenerator("valid", src_lang, trg_lang)

    train_iter = Multi30k(split='train', language_pair=(src_lang, trg_lang))
    valid_iter = Multi30k(split='valid', language_pair=(src_lang, trg_lang))
    train_dataloader = DataLoader(train_iter, batch_size=batch_size, collate_fn=train_generator.collate_fn)    
    valid_dataloader = DataLoader(valid_iter, batch_size=batch_size, collate_fn=valid_generator.collate_fn)

    train_src_vocabs, valid_src_vocabs = train_generator.vocab_transform[src_lang].get_itos(), valid_generator.vocab_transform[src_lang].get_itos()
    train_trg_vocabs, valid_trg_vocabs = train_generator.vocab_transform[trg_lang].get_itos(), valid_generator.vocab_transform[trg_lang].get_itos()
    src_pad_idx, trg_pad_idx, trg_sos_idx = train_generator.PAD_IDX, train_generator.PAD_IDX, train_generator.SOS_IDX

    model = Transformer(src_pad_idx=src_pad_idx,
                        trg_pad_idx=trg_pad_idx,
                        trg_sos_idx=trg_sos_idx,
                        d_model=d_model,
                        encoder_vocab_size=len(train_src_vocabs),
                        decoder_vocab_size=len(train_trg_vocabs),
                        max_len=max_len,
                        ffn_hidden=ffn_hidden,
                        num_head=n_heads,
                        num_layers=n_layers,
                        drop_prob=drop_prob,
                        device=device).to(device)

    print(f'The model has {count_parameters(model):,} trainable parameters')
    model.apply(initialize_weights)
    optimizer = Adam(params=model.parameters(),
                    lr=init_lr,
                    weight_decay=weight_decay,
                    eps=adam_eps)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                    verbose=True,
                                                    factor=factor,
                                                    patience=patience)

    criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)
    train_losses, test_losses, bleus = [], [], []
    for epoch in range(total_epoch):
        train_loss = train(model, train_dataloader, optimizer, criterion, clip)
        valid_loss, bleu = evaluate(model, valid_dataloader, valid_trg_vocabs, criterion)

        if epoch > warmup:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)

        if valid_loss < best_loss:
            best_loss = valid_loss
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            torch.save(model.state_dict(), f'{save_dir}/best_epoch_{epoch}.pt')

        f = open(f'{save_dir}/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()

        f = open(f'{save_dir}/bleu.txt', 'w')
        f.write(str(bleus))
        f.close()

        f = open(f'{save_dir}/test_loss.txt', 'w')
        f.write(str(test_losses))
        f.close()

        print(f'Epoch: {epoch + 1}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')


if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)
