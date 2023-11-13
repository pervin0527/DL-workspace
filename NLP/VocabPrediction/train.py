import math
import torch
import torch.nn as nn

from tqdm import tqdm
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from models.model import RNN
from utils.util import make_dir, read_file
from data.wikitext_dataset import download_wikitext

def tokenize_file(text):
    """
    데이터셋에 포함된 각각의 문장들을 토큰화(문장을 단어 단위로 분리)하며, 마지막에 <eos> 토큰을 추가한다.
    """
    tokenized_data = []
    for line in text:
        tokens = tokenizer(line.strip()) + ['<eos>']
        tokenized_data.append(tokens)
    return tokenized_data


def build_vocab(data_tokens):
    """
    토큰화된 데이터셋을 이용해 단어 사전을 생성한다.
    """
    return build_vocab_from_iterator(data_tokens, specials=['<unk>', '<eos>'], min_freq=3)


def get_data(tokenized_data, vocab, batch_size):
    """
    토큰들이 Vocab내 index로 mapping하고 리스트에 저장.
    결과적으로 전체 문장들이 하나의 문장으로 모두 연결된다.

    데이터의 총 원소 수(numel())를 batch_size로 나누어 전체 데이터를 몇 개의 배치로 나눌 수 있는지 계산한다. 
    이를 통해, 데이터의 길이를 배치 크기에 맞게 조정합니다. 즉, 모든 배치의 길이를 통일합니다.

    ex) 전체 데이터의 수가 2086708일 때, batch_size=128 ---> 2086708 // 128 = 16302 따라서 num_batches = 16302
    """
    data = []
    for tokens in tokenized_data:
        token_indices = [vocab[token] for token in tokens]
        data.extend(token_indices)

    data = torch.LongTensor(data) ## [2086708]
    num_batches = data.numel() // batch_size ## 16302
    data = data[:num_batches * batch_size] ## 2086656. 남은 52개의 단어는 제외하여 모든 데이터들의 길이를 통일시킨다.
    data = data.view(batch_size, -1) ## [batch_size, 16302]로 reshape
    
    return data


def get_batch(data, seq_len, idx):
    """
    idx값부터 idx + seq_len까지의 길이로 slicing을 하게 되는데,
    - src는 처음부터 마지막 단어(seq_len)이전까지. 즉, 마지막 단어를 포함하지 않는다.
    - target은 첫 단어를 제외한 마지막 단어를 포함한다.
    """
    src = data[:, idx:idx+seq_len]                   
    target = data[:, idx+1:idx+seq_len+1]
    
    return src, target


def train(model, data, optimizer, criterion, batch_size, seq_len, clip, device):
    epoch_loss = 0
    model.train()

    """
    특정 시퀀스 길이(seq_len)에 대해 데이터를 추가로 조정하기 위한 과정으로, 모든 배치가 지정된 시퀀스 길이에 정확히 맞도록 하여, 모델이 일관된 길이의 시퀀스를 처리하도록 한다.
    """
    num_batches = data.shape[-1] ## 16302
    data = data[:, :num_batches - (num_batches -1) % seq_len] ## [128, 16301]
    num_batches = data.shape[-1] ## 16301

    hidden = model.init_hidden(batch_size, device)
    
    for idx in tqdm(range(0, num_batches - 1, seq_len), desc='Training: ',leave=False):
        optimizer.zero_grad()
        hidden = model.detach_hidden(hidden)

        src, target = get_batch(data, seq_len, idx) ## [128, 50], [128, 50]
        src, target = src.to(device), target.to(device)
        batch_size = src.shape[0]
        prediction, hidden = model(src, hidden)

        prediction = prediction.reshape(batch_size * seq_len, -1)   
        target = target.reshape(-1)
        loss = criterion(prediction, target)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item() * seq_len

    return epoch_loss / num_batches


def evaluate(model, data, criterion, batch_size, seq_len, device):
    epoch_loss = 0
    model.eval()
    num_batches = data.shape[-1]
    data = data[:, :num_batches - (num_batches -1) % seq_len]
    num_batches = data.shape[-1]

    hidden = model.init_hidden(batch_size, device)

    with torch.no_grad():
        for idx in range(0, num_batches - 1, seq_len):
            hidden = model.detach_hidden(hidden)
            src, target = get_batch(data, seq_len, idx)
            src, target = src.to(device), target.to(device)
            batch_size= src.shape[0]

            prediction, hidden = model(src, hidden)
            prediction = prediction.reshape(batch_size * seq_len, -1)
            target = target.reshape(-1)

            loss = criterion(prediction, target)
            epoch_loss += loss.item() * seq_len
    return epoch_loss / num_batches


def generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    tokens = tokenizer(prompt)
    indices = [vocab[t] for t in tokens]
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)
            prediction = torch.multinomial(probs, num_samples=1).item()
            
            while prediction == vocab['<unk>']:
                prediction = torch.multinomial(probs, num_samples=1).item()

            if prediction == vocab['<eos>']:
                break

            indices.append(prediction)

    itos = vocab.get_itos()
    tokens = [itos[i] for i in indices]
    return tokens


if __name__ == "__main__":
    data_dir = "/home/pervinco/Datasets/wikitext"
    save_dir = "/home/pervinco/Models/wikitext"

    batch_size = 128
    epochs = 100
    learning_rate = 0.001
    
    embedding_dim = 1024
    hidden_dim = 1024
    num_layers = 2
    dropout_rate = 0.65

    tie_weights = True
    sequence_length = 50
    clip = 0.25

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    make_dir(save_dir)

    ## Download Dataset
    download_wikitext(data_dir)

    ## Define Dataset Dir
    data_dir = f"{data_dir}/wikitext-2"
    train_file = f"{data_dir}/wiki.train.tokens"
    valid_file = f"{data_dir}/wiki.valid.tokens"
    test_file = f"{data_dir}/wiki.test.tokens"

    ## Read File
    train_text = read_file(train_file)
    valid_text = read_file(valid_file)
    test_text = read_file(test_file)

    ## Tokenize
    tokenizer = get_tokenizer('basic_english')
    train_data_tokens = tokenize_file(train_text)
    valid_data_tokens = tokenize_file(valid_text)
    test_data_tokens = tokenize_file(test_text)

    ## 단어집 생성.
    vocab = build_vocab(train_data_tokens + valid_data_tokens + test_data_tokens)
    vocab.set_default_index(vocab['<unk>'])
    vocab_size = len(vocab)
    print(vocab_size) ## 28783

    # 전체 토큰의 수, 전체 batch 단위의 수
    train_data = get_data(train_data_tokens, vocab, batch_size) ## data.numel() : 2086708, num_batches : 16302, data : 218177
    valid_data = get_data(valid_data_tokens, vocab, batch_size) 
    test_data = get_data(test_data_tokens, vocab, batch_size)

    model = RNN(vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate, tie_weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=0)

    best_valid_loss = float('inf')
    for epoch in range(epochs):
        train_loss = train(model, train_data, optimizer, criterion, batch_size, sequence_length, clip, device)
        valid_loss = evaluate(model, valid_data, criterion, batch_size, sequence_length, device)
        
        lr_scheduler.step(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'{save_dir}/best.pt')

        print(f'\tTrain Perplexity: {math.exp(train_loss):.3f}')
        print(f'\tValid Perplexity: {math.exp(valid_loss):.3f}')

    prompt = 'Think about'
    max_seq_len = 30
    seed = 0

    # convert the code above into a for loop
    temperatures = [0.5, 0.7, 0.75, 0.8, 1.0]
    for temperature in temperatures:
        generation = generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed)
        print(str(temperature)+'\n'+' '.join(generation)+'\n')