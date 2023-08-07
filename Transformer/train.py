import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k
from data.en_de import TokenGenerator
from utils.data_utils import get_input_gt, get_masks
from models.transformer import Transformer

torch.set_printoptions(profile="full")

def train(model, optimizer):
    model.train()
    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TRG_LANGUAGE))
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=token_generator.collate_fn)
    
    total_loss = 0
    for src, trg in train_dataloader:
        src, trg = src.to(DEVICE), trg.to(DEVICE)
        trg_input, trg_out = get_input_gt(trg)

        src_mask, trg_mask, src_tokens, trg_tokens = get_masks(src, trg_input, pad_token_id)
        pred = model(src, trg_input, src_mask, trg_mask)  
        
        optimizer.zero_grad()
        # loss = loss_func(pred.reshape(-1, pred.shape[-1]), trg_out.reshape(-1))
        loss = loss_func(pred.reshape(-1, target_vocab_size), trg_out.reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(list(train_dataloader))


def evaluate(model):
    model.eval()
    val_iter = Multi30k(split="valid", language_pair=(SRC_LANGUAGE, TRG_LANGUAGE))
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=token_generator.collate_fn)

    total_loss = 0
    for src, trg in val_dataloader:
        src, trg = src.to(DEVICE), trg.to(DEVICE)
        trg_input, trg_out = get_input_gt(trg)

        src_mask, trg_mask, src_tokens, trg_tokens = get_masks(src, trg_input, pad_token_id)
        pred = model(src, trg_input, src_mask, trg_mask)
        loss = loss_func(pred.reshape(-1, target_vocab_size), trg_out.reshape(-1))
        total_loss += loss.item()

    return total_loss / len(list(val_dataloader))


if __name__ == "__main__":
    D_MODEL = 512
    NUM_HEADS = 8
    NUM_LAYERS = 6
    FFN_DIM = 512
    MAX_SEQ_LEN = 512
    DROP_PROB = 0.1

    SRC_LANGUAGE = 'de'
    TRG_LANGUAGE = 'en'
    SAVE_DIR = "/home/pervinco/Models/Transformer"
    EPOCHS = 1000
    BATCH_SIZE = 128
    LR = 0.0001
    BETAS = (0.9, 0.98)
    EPSILON = 1e-9
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    token_generator = TokenGenerator(split="train", source_lang=SRC_LANGUAGE, target_lang=TRG_LANGUAGE)
    source_vocab_size = len(token_generator.vocab_transform[SRC_LANGUAGE])
    target_vocab_size = len(token_generator.vocab_transform[TRG_LANGUAGE])
    pad_token_id = token_generator.PAD_IDX
    print(f"SRC VOCABS : {source_vocab_size}, TRG VOCABS : {target_vocab_size}, PAD TOKENS : {pad_token_id}")

    model = Transformer(src_vocab_size=source_vocab_size,
                        trg_vocab_size=target_vocab_size,
                        d_model=D_MODEL,
                        num_heads=NUM_HEADS,
                        num_layers=NUM_LAYERS,
                        max_seq_len=MAX_SEQ_LEN,
                        drop_prob=DROP_PROB).to(DEVICE)
    
    loss_func = nn.NLLLoss(ignore_index=pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=BETAS, eps=EPSILON)

    best_val_loss = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, optimizer)
        valid_loss = evaluate(model)
        print(f"Epoch : {epoch}, Train loss : {train_loss:.4f}, Valid loss : {valid_loss:.4f}")

        if epoch == 1:
            best_val_loss = valid_loss
        
        if epoch > 1:
            if best_val_loss > valid_loss:
                best_val_loss = valid_loss

                torch.save(model.state_dict(), f"{SAVE_DIR}/best-{epoch}epoch.pt")