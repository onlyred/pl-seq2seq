import torch
import torch.nn as nn
import torch.nn.functional as F

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


class AttnEncoder(nn.Module):
    def __init__(self, input_dim:int, enc_hidden_dim:int, dec_hidden_dim:int, 
                 embed_dim:int, nlayers:int, drop_rate:float):
        super().__init__()
        bidirect = True
        bi_dim = 2
        self.nlayers   = nlayers

        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.lstm      = nn.LSTM(embed_dim, enc_hidden_dim, nlayers, bidirectional=bidirect, dropout=drop_rate)
        self.fc        = nn.Linear(enc_hidden_dim * bi_dim, dec_hidden_dim)
        self.dropout   = nn.Dropout(drop_rate)

    def forward(self, x):
        '''
        x = [seq_len, batch_size]
        '''
        embedded = self.dropout( self.embedding(x) ) # [seq_len, batch_size, embed_dim]
        outputs, (h, c) = self.lstm(embedded)
        # outputs = [seq_len, batch_size, en_hidden_dim * num direction]
        # h       = [nlayers * num directions, batch_size, hidden_dim]

        #hidden (bidirectional) is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #outputs are always from the last layer
        
        #hidden [-2, :, : ] is the last of the forwards RNN 
        #hidden [-1, :, : ] is the last of the backwards RNN

        #initial decoder hidden is final hidden state of the forwards and backwards 
        #  encoder RNNs fed through a linear layer
        for i in range(self.nlayers):
            f_h, b_h = h[i*2,:,:],h[i*2+1,:,:]
            f_c, b_c = c[i*2,:,:],c[i*2+1,:,:]
            if i == 0:
                h_out = torch.tanh(self.fc(torch.cat((f_h, b_h), dim=1))).unsqueeze(0)
                c_out = torch.tanh(self.fc(torch.cat((f_c, b_c), dim=1))).unsqueeze(0)
            else:
                h_temp  = torch.tanh(self.fc(torch.cat((f_h, b_h), dim=1))).unsqueeze(0)
                c_temp  = torch.tanh(self.fc(torch.cat((f_c, b_c), dim=1))).unsqueeze(0)

                h_out = torch.cat((h_out, h_temp), dim=0)
                c_out = torch.cat((c_out, c_temp), dim=0)
        # h_out  = [nlayer, batch_size, dec_hidden_dim]
        # c_out  = [nlayer, batch_size, dec_hidden_dim]
        return outputs, h_out, c_out

class Attention(nn.Module):
    def __init__(self, enc_hidden_dim:int, dec_hidden_dim:int):
        super().__init__()
        bi_dim    = 2
        self.attn = nn.Linear((enc_hidden_dim * bi_dim) + dec_hidden_dim, dec_hidden_dim)
        self.v    = nn.Linear(dec_hidden_dim, 1, bias=False)

    def forward(self, x, h):
        '''
        x (encoder_outputs) = [seq_len, batch_size, enc_hidden_dim * 2]
        h (hidden) = [batch_size, dec_hidden_dim]
        '''
        batch_size = x.size(1)
        seq_len    = x.size(0)

        # use last hidden layer
        # repeate decoder hidden state seq_len times
        h = h[-1].unsqueeze(1).repeat(1, seq_len, 1) # [batch_size, seq_len, dec_hidden_dim]
        x = x.permute(1,0,2)   # [batch_size, seq_len, enc_hidden_dim * 2]

        energy = torch.tanh(self.attn(torch.cat((h, x), dim=2))) # [batch_size, seq_len, dec_hidden_dim]
        attn = self.v(energy).squeeze(2) # [batch_size, seq_len]
        return F.softmax(attn, dim=1)

class AttnDecoder(nn.Module):
    def __init__(self, output_dim:int, enc_hidden_dim:int, dec_hidden_dim:int, 
                 embed_dim:int, nlayers:int, drop_rate:float, attn:object):
        super().__init__()
        bi_dim              = 2
        self.output_dim     = output_dim
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.embed_dim      = embed_dim
        self.nlayers        = nlayers
        self.attention      = attn

        self.embedding      = nn.Embedding(output_dim, embed_dim)
        self.lstm           = nn.LSTM((enc_hidden_dim * bi_dim) + embed_dim, dec_hidden_dim, nlayers, 
                                      dropout=drop_rate)
        self.fc             = nn.Linear((enc_hidden_dim * bi_dim) + dec_hidden_dim + embed_dim,
                                        output_dim)
        self.dropout        = nn.Dropout(drop_rate)

    def forward(self, x, h, c, encoder_outputs):
        '''
        x = [batch_size]
        h = [nlayer, batch_size, dec_hidden_dim]
        c = [nlayer, batch_size, dec_hidden_dim]
        encoder_outputs = [seq_len, batch_size, enc_hidden_dim * 2]
        '''
        x = x.unsqueeze(0) # [1, batch_size]
        embedded = self.dropout(self.embedding(x)) # [1, batch_size, embed_dim]
        a = self.attention(encoder_outputs, h) # [batch_size, seq_len]

        a = a.unsqueeze(1) # [batch_size, 1, seq_len]
        encoder_outputs = encoder_outputs.permute(1,0,2) # [batch_size, seq_len, enc_hidden_dim * 2]
        weighted = torch.bmm(a, encoder_outputs) # [batch_size, 1, enc_hidden_dim * 2]
        weighted = weighted.permute(1,0,2) # [1, batch_size, enc_hidden_dim * 2]

        rnn_input = torch.cat((embedded, weighted), dim=2) # [1, batch_size, (enc_hidden_dim * 2) + embed_dim]
        output, (h, c) = self.lstm(rnn_input, (h, c))
        # output = [seq_len, batch_size, dec_hidden_dim * n directions]
        # h      = [nlayers * n directions, batch_size, dec_hidden_dim]

        embedded = embedded.squeeze(0)
        output   = output.squeeze(0)
        weighted = weighted.squeeze(0)

        p = self.fc(torch.cat((output, weighted, embedded), dim=1)) # [batch_size, output_dim]
        return p, h.squeeze(0), c.squeeze(0)
