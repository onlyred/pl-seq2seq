import argparse
import random
import pytorch_lightning as pl
import torch
import torch.optim as optim
import torch.nn as nn

from models import AttnEncoder, AttnDecoder, Attention
from utils import Tokens, IDs

class Seq2Seq_With_Attn(pl.LightningModule):
    def __init__(self, hparams):
        super(Seq2Seq_With_Attn,self).__init__()
        self.save_hyperparameters() # for checkpoint

        self.input_dim  = hparams.input_dim
        self.enc_hidden = hparams.enc_hidden
        self.dec_hidden = hparams.dec_hidden
        self.output_dim = hparams.output_dim
        self.embed_dim  = hparams.embed_dim
        self.nlayers    = hparams.nlayers
        self.drop_rate  = hparams.drop_rate
        self.epochs     = hparams.epochs
        self.lr         = hparams.lr
        self.force_teach= hparams.force_teach
        self.batch_size = hparams.batch_size
 
        self.criterion  = nn.CrossEntropyLoss(ignore_index=IDs.PAD.value)

        self.encoder = AttnEncoder(self.input_dim, self.enc_hidden, self.dec_hidden,
                                   self.embed_dim, self.nlayers, self.drop_rate)
        self.attn    = Attention(self.enc_hidden, self.dec_hidden)
        self.decoder = AttnDecoder(self.output_dim, self.enc_hidden, self.dec_hidden,
                                   self.embed_dim, self.nlayers, self.drop_rate, self.attn)

    def forward(self, x, y):
        '''
        x = [src_seq_len, batch_size]
        y = [tgt_seq_len, batch_size]
        '''
        y_len   = y.size(0)
        batch_size = y.size(1)

        outputs = torch.zeros(y_len, batch_size, self.output_dim).to(self.device)
        
        enc_outputs, h, c = self.encoder(x)
        # first input to the decoder is the <bos> tokens
        y_in    = y[0,:]    # input <eos> tokens
        for i in range(1, y_len):
            # insert input token embedding, previous hidden and previous cell states
            p, h, c = self.decoder(y_in, h, c, enc_outputs)
            outputs[i] = p
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < self.force_teach
            # get the highest predicted token from our predictions
            top1 = p.argmax(1)  
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            y_in = y[i] if teacher_force else top1
        return outputs

    def training_step(self, batch, batch_nb):
        x, y= batch
        batch_size = y.size(1)
        output = self.forward(x, y)
        output = output[1:].view(-1, self.output_dim)
        target = y[1:].view(-1)
        loss   = self.criterion(output, target)

        self.log('tr_loss', loss, prog_bar=True, on_step=True, batch_size=batch_size)
        return {'loss' : loss }

    def validation_step(self, batch, batch_nb):
        x, y = batch
        batch_size = y.size(1)
        output = self.forward(x, y)
        # y = [ tgt_len, batch_size ]
        # output = [ tgt_len, batch_size, output_dim ]
        output = output[1:].view(-1, self.output_dim)
        target = y[1:].view(-1)
        loss   = self.criterion(output, target)

        self.log('val_loss', loss, prog_bar=True, batch_size=batch_size)
        return {'val_loss' : loss}

    def validation_end(self, outputs):  # optional
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss)
        return {'avg_val_loss' : avg_loss}

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_end(self, outputs):
        return self.validation_end(outputs)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, 
                                                lr_lambda=lambda epoch : 0.95 ** self.epochs)
        return {"optimizer" : optimizer, "lr_scheduler" : scheduler}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--input_dim', type=int, default=1)
        parser.add_argument('--enc_hidden', type=int, default=1)
        parser.add_argument('--dec_hidden', type=int, default=1)
        parser.add_argument('--output_dim', type=int, default=1)
        parser.add_argument('--embed_dim', type=int, default=1)
        parser.add_argument('--nlayers', type=int, default=1)
        parser.add_argument('--drop_rate', type=float, default=0.2)
        parser.add_argument('--force_teach', type=float, default=0.5)
        return parser
