import torch

from tqdm import tqdm
from torch import nn, optim

from config import *
from data.dataset import Multi30kDataset
from models.build_model import build_model
from utils import get_bleu_score, greedy_decode
from data.util import download_multi30k, make_cache


def train(model, data_loader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for (src, trg) in tqdm(data_loader, desc="train", leave=False):
        src = src.to(DEVICE)
        trg = trg.to(DEVICE)
        trg_x = trg[:, :-1]
        trg_y = trg[:, 1:]

        optimizer.zero_grad()

        output, _ = model(src, trg_x)

        y_hat = output.contiguous().view(-1, output.shape[-1])
        y_gt = trg_y.contiguous().view(-1)
        loss = criterion(y_hat, y_gt)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(list(data_loader))


def evaluate(model, data_loader, criterion):
    model.eval()
    epoch_loss = 0

    total_bleu = []
    with torch.no_grad():
        for (src, trg) in tqdm(data_loader, desc="eval", leave=False):
            src = src.to(DEVICE)
            trg = trg.to(DEVICE)
            trg_x = trg[:, :-1]
            trg_y = trg[:, 1:]

            output, _ = model(src, trg_x)

            y_hat = output.contiguous().view(-1, output.shape[-1])
            y_gt = trg_y.contiguous().view(-1)
            loss = criterion(y_hat, y_gt)

            epoch_loss += loss.item()
            score = get_bleu_score(output, trg_y, DATASET.trg_vocab, DATASET.SPECIALS)
            total_bleu.append(score)

    loss_avr = epoch_loss / len(list(data_loader))
    bleu_score = sum(total_bleu) / len(total_bleu)

    return loss_avr, bleu_score


def initialize_weights(model):
    if hasattr(model, 'weight') and model.weight.dim() > 1:
        nn.init.kaiming_uniform_(model.weight.data)


def main():
    train_iter, valid_iter, test_iter = DATASET.get_iter(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    model = build_model(len(DATASET.src_vocab), len(DATASET.trg_vocab), device=DEVICE, drop_prob=DROP_PROB)
    model.apply(initialize_weights)

    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, eps=ADAM_EPS)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, verbose=True, factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)
    criterion = nn.CrossEntropyLoss(ignore_index=DATASET.PAD_IDX)

    print("\nTrain Start")
    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR, exist_ok=True)

    min_val_loss = 0
    for epoch in range(EPOCHS):
        train_loss = train(model, train_iter, optimizer, criterion)
        valid_loss, bleu_score  = evaluate(model, valid_iter, criterion)

        if epoch == 0:
            min_val_loss = valid_loss

        if epoch > 1:
            if valid_loss < min_val_loss:
                min_val_loss = valid_loss
                ckpt = f"{SAVE_DIR}/{epoch:04}.pt"
                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'train_loss': train_loss,
                            'val_loss' : valid_loss}, ckpt)

        if epoch > WARM_UP_STEP:
            scheduler.step(valid_loss)

        print(f"Epoch : {epoch + 1} | train_loss: {train_loss:.5f} valid_loss: {valid_loss:.5f}, bleu_score: {bleu_score:.5f}")
        print("Predict : ", DATASET.translate(model, "A little girl climbing into a wooden playhouse .", greedy_decode))
        print(f"Answer : Ein kleines Mädchen klettert in ein Spielhaus aus Holz . \n")

    test_loss, bleu_score = evaluate(model, test_iter, criterion)
    print(f"test_loss: {test_loss:.5f}")
    print(f"bleu_score: {bleu_score:.5f}")


if __name__ == "__main__":
    download_multi30k(DATA_DIR)
    make_cache(f"{DATA_DIR}/Multi30k")

    DATASET = Multi30kDataset(data_dir=f"{DATA_DIR}/Multi30k", source_language=SRC_LANGUAGE,  target_language=TRG_LANGUAGE,  max_seq_len=MAX_SEQ_LEN, vocab_min_freq=2)
    
    main()