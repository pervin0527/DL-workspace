import torch

def get_input_gt(trg):
    trg_input = trg[:, :-1]
    # trg_out = trg[:, 1:].reshape(-1, 1)
    trg_out = trg[:, 1:]

    return trg_input, trg_out

def get_masks(src, trg, pad_token_id):
    batch_size = src.size(0)
    device = trg.device

    src_mask = (src != pad_token_id).view(batch_size, 1, 1, -1)
    src_tokens = torch.sum(src_mask, dtype=torch.int64)

    sequence_length = trg.shape[1]
    trg_padding_mask = (trg != pad_token_id).view(batch_size, 1, 1, -1)
    trg_forward_mask = torch.triu(torch.ones((1, 1, sequence_length, sequence_length), device=device) == 1).transpose(2, 3)

    trg_mask = trg_padding_mask & trg_forward_mask
    trg_tokens = torch.sum(trg_padding_mask, dtype=torch.int64)

    return src_mask, trg_mask, src_tokens, trg_tokens