import random
import torch
import torch.nn as nn

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.01)

class Encoder(nn.Module):
    def __init__(self, input_dim, embedd_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_dim, embedd_dim)
        self.rnn = nn.LSTM(embedd_dim, hidden_dim, num_layers, dropout = dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        """
        src : [src_length, batch_size]
        embedded : [src_length, batch_size, embedd_dim]
        outputs : [src_len, batch_size, hidden_dim * num_directions]
        hidden : [num_layers * num_directions, batch_size, hidden_dim] 
        cell : [num_layers * num_directions, batch_size, hidden_dim]
        """
        
        embedded = self.dropout(self.embedding(src))        
        outputs, (hidden, cell) = self.rnn(embedded)
        
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, embedd_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(output_dim, embedd_dim)
        self.rnn = nn.LSTM(embedd_dim, hidden_dim, num_layers, dropout = dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        """
        input : <sos> token in target tensor.
        context : [num_layers, batch_size, hidden_dim]

        output : [seq_len, batch_Size, hidden_dim * num_directions]
        hidden : [num_layers * num_directions, batch_size, hidden_dim]
        cell : [num_layers * num_directions, batch_size, hidden_dim]
        """
        input = input.unsqueeze(0) ## [batch_size] -> [1, batch_size]
        embedded = self.dropout(self.embedding(input)) ## embedded = [1, batch_size, emb_dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        prediction = self.fc_out(output.squeeze(0)) ## prediction = [batch_size, output_dim]
        
        return prediction, hidden, cell
    

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hidden_dim == decoder.hidden_dim, "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.num_layers == decoder.num_layers, "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        src : [src_length, batch_size]
        trg : [trg_length, batch_size]
        outputs : tensor to store decoder outputs
        """
        batch_size, trg_len = trg.shape[1], trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)        
        hidden, cell = self.encoder(src) ## last hidden state of encoder -> used as the initial hidden state of the decoder.
        
        input = trg[0,:] ## first input to the decoder is the <sos> tokens
        for t in range(1, trg_len):
            ## insert input token embedding, previous hidden and previous cell states
            ## receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell) ## output : [batch_size, seq_len]
            
            ## place predictions in a tensor holding predictions for each token
            outputs[t] = output
            # output = output.squeeze(0)
            
            ## decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            ## get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            ## if not, use predicted token
            ## if teacher forcing, use actual next token as next input
            input = trg[t] if teacher_force else top1
        
        return outputs
    

class EncoderGRU(nn.Module):
    def __init__(self, input_dim, embedd_dim, hidden_dim, dropout):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(input_dim, embedd_dim)
        self.rnn = nn.GRU(embedd_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden_state = self.rnn(embedded)

        return hidden_state

class DecoderGRU(nn.Module):
    def __init__(self, output_dim, embedd_dim, hidden_dim, dropout):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, embedd_dim)
        self.rnn = nn.GRU(embedd_dim + hidden_dim, hidden_dim)
        self.out = nn.Linear(embedd_dim + hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, context):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))

        concat = torch.cat((embedded, context), dim=2)
        output, hidden = self.rnn(concat, hidden)

        output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)), dim=1)
        prediction = self.out(output)

        return prediction, hidden
    

class Seq2SeqGRU(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hidden_dim == decoder.hidden_dim, "Hidden dimensions of encoder and decoder must be equal!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):       
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)        
        context = self.encoder(src)        
        hidden = context
        
        input = trg[0,:]
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, context)
            outputs[t] = output
            
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1) 
            input = trg[t] if teacher_force else top1

        return outputs