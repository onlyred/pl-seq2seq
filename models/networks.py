import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, embed_dim:int, 
                 nlayers:int, bidirect:bool, drop_rate:float):
        super().__init__()

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.embed_dim  = embed_dim
        self.nlayers    = nlayers
        if bidirect:
            self.bi_dim = 2
        else:
            self.bi_dim = 1

        self.embedding = nn.Embedding(input_dim, embed_dim)

        self.lstm      = nn.LSTM(embed_dim, hidden_dim, nlayers, bidirectional=bidirect, dropout=drop_rate)
        self.dropout   = nn.Dropout(drop_rate)

    def forward(self, x):
        '''
        input
            > x : input data [seq_len, batch_size]
        output
            > outputs : [seq_len, batch_size, hidden_dim * n directions]
            > h       : [nlayers * n directions, batch_size, hidden_dim]
            > c       : [nlayers * n directions, batch_size, hidden_dim]    
        '''
        batch_size = x.size(1)
        embeded = self.dropout( self.embedding(x) )   # [ seq_len, batch_size, embed_dim ]
        outputs, (h, c) = self.lstm(embeded)
        return h, c

class Decoder(nn.Module):
    def __init__(self, output_dim:int, hidden_dim:int, embed_dim:int, 
                 nlayers:int, bidirect:bool, drop_rate:float):
        super().__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.nlayers    = nlayers
        if bidirect:
            self.bi_dim = 2
        else:
            self.bi_dim = 1

        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, nlayers, bidirectional=bidirect, dropout=drop_rate)
        self.fc        = nn.Linear(hidden_dim * self.bi_dim, output_dim)
        self.dropout   = nn.Dropout(drop_rate)

    def forward(self, x, h, c):
        '''
        input
            > x  : input data [batch_size]
            > h  : hidden state [nlayers* n directions, batch_size, hidden_dim] 
            > c  : cell state [nlayers* n directions, batch_size, hidden_dim] 
        output
            > pred : prediction [batch_size, output_dim]
            > h    : hidden state [nlayers, batch_size, hidden_dim]
            > c    : cell state [nlayers, batch_size, hidden_dim]
        ** n direction in the decoder will both always be 1.
        '''
        x = x.unsqueeze(0)  # [1, batch_size]
        embeded = self.dropout( self.embedding(x) ) # [1, batch_size, embed_dim]
        output, (h, c) = self.lstm(embeded, (h, c)) # [seq_len, batch_size, hidden_dim * n directions]
        pred = self.fc(output.squeeze(0))
        return pred, h, c
