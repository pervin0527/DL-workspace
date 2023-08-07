import csv
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torch.autograd import Variable

def load_csv(file_path):
    print(f'Load Data | file path: {file_path}')
    with open(file_path, 'r') as csv_file:
       csv_reader = csv.reader(csv_file)
       
       lines = []
       for line in csv_reader:
        line[0] = line[0].replace(';','')
        lines.append(line)
    print(f'Load Complete | file path: {file_path}')
    
    return lines

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class TranslationDataset(Dataset):
    def __init__(self, tokenizer:BertTokenizer, file_path, max_length):
        pad_token_idx = tokenizer.pad_token_id
        csv_data = load_csv(file_path)

        self.docs = []
        for line in csv_data:
            input = tokenizer.encode(line[0], max_length=max_length, truncation=True)
            rest = max_length - len(input)
            input = torch.tensor(input + [pad_token_idx] * rest)

            target = tokenizer.encode(line[1], max_length=max_length, truncation=True)
            rest = max_length - len(target)
            target = torch.tensor(target + [pad_token_idx] * rest)

            doc = {'input_str' : tokenizer.convert_ids_to_tokens(input),
                   'input' : input,
                   'input_mask' : (input != pad_token_idx).unsqueeze(-2),
                   'target_str' : tokenizer.convert_ids_to_tokens(target),
                   'target' : target,
                   'target_mask' : self.make_std_mask(target, pad_token_idx),
                   'token_num': (target[...,1:] != pad_token_idx).data.sum()}
            self.docs.append(doc)

    @staticmethod
    def make_std_mask(trg, pad_token_idx):
        target_mask = (trg != pad_token_idx).unsqueeze(-2)
        target_mask = target_mask & Variable(subsequent_mask(trg.size(-1)).type_as(target_mask.data))

        return target_mask.squeeze()

    def __len__(self):
        return len(self.docs)
    
    def __getitem__(self, idx):
        item = self.docs[idx]
        return item
    
if __name__ == "__main__":
    vocab_path = "/home/pervinco/Desktop/pytorch-transformer/data/wiki-vocab.txt"
    data_path = "/home/pervinco/Desktop/pytorch-transformer/data/test.csv"
    max_length = 512

    tokenizer = BertTokenizer(vocab_file=vocab_path, do_lower_case=False)
    print(tokenizer)

    dataset = TranslationDataset(tokenizer=tokenizer, file_path=data_path, max_length=max_length)
    for data in dataset:
        print(data['input_str'], '\n')
        print(data['input'], '\n')
        print(data['input_mask'], '\n')

        print(data['target_str'], '\n')
        print(data['target'], '\n')
        print(data['target_mask'], '\n')
        
        print(data['token_num'])

        break