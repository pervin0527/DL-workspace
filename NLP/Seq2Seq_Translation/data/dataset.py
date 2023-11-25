import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class TranslationDataset(Dataset):
    def __init__(self, dataset):
        self.src_indices, self.trg_indices = dataset[0], dataset[1]

    def __len__(self):
        return len(self.src_indices)

    def __getitem__(self, idx):
        src = torch.tensor(self.src_indices[idx], dtype=torch.long)
        trg = torch.tensor(self.trg_indices[idx], dtype=torch.long)
        
        return src, trg
    
    def collate_fn(self, batch):
        src_batch, trg_batch = zip(*batch)
        src_batch_padded = pad_sequence(src_batch, batch_first=True, padding_value=0) 
        trg_batch_padded = pad_sequence(trg_batch, batch_first=True, padding_value=0)

        src_batch_padded = src_batch_padded.view(-1, src_batch_padded.shape[0])
        trg_batch_padded = trg_batch_padded.view(-1, trg_batch_padded.shape[0])
        
        return src_batch_padded, trg_batch_padded