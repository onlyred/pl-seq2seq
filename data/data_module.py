import sys, os
import argparse
import torch
import spacy
import pytorch_lightning as pl
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import Multi30k
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from typing import Iterable, List

from utils import IDs, Tokens
'''
torchtext > 1.0
'''
class Multi30k_DataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()
        self.SRC_LANG   ='de'
        self.TGT_LANG   ='en'

        self.batch_size = hparams.batch_size

    def prepare_data(self):
        '''
        download spacy-models (de_core_new_sm, en_core_web_sm)
        '''
        #os.system("python -m spacy download en_core_web_sm")
        #os.system("python -m spacy download de_core_news_sm")

    def setup(self, stage: str = None):
        if stage in (None, "fit"):
            self.train_iter, self.valid_iter = Multi30k(split=('train','valid'), 
                                               language_pair=(self.SRC_LANG, self.TGT_LANG))
        elif stage in (None, "test"):
            self.train_iter, self.test_iter = Multi30k(split=('train','test'), 
                                               language_pair=(self.SRC_LANG, self.TGT_LANG))
        else:
            raise NotImplementedError
        self._preprocess(self.train_iter)

    def _preprocess(self, data_iter: Iterable):
        def sequential_transforms(*transforms):
            '''
            Helper function for binding sequential jobs
            '''
            def func(txt_input):
                for transform in transforms:
                    txt_input = transform(txt_input)
                return txt_input
            return func

        def tensor_transform(token_ids: List[int]):
            '''
            Function for adding 'bos'/'eos' and making tensor about index of input sequence
            '''
            return torch.cat((torch.tensor([IDs.BOS.value]),
                              torch.tensor(token_ids),
                              torch.tensor([IDs.EOS.value])))

        token_transform = {}
        vocab_transform = {}
        text_transform  = {}
        LANGUAGES = [self.SRC_LANG, self.TGT_LANG]
        # Generate tokenizer for SRC, TGT
        token_transform[self.SRC_LANG] = get_tokenizer('spacy', language='de_core_news_sm')
        token_transform[self.TGT_LANG] = get_tokenizer('spacy', language='en_core_web_sm')
        def yield_tokens(data_iter: Iterable, language: str)->List[str]:
            '''
            Helper function for making token list
            '''
            language_index= {self.SRC_LANG:0, self.TGT_LANG:1}
            for data_sample in data_iter:
                yield token_transform[language](data_sample[language_index[language]])
        # Define special symbols and indexes
        special_symbols = {}
        for t, i in zip(Tokens, IDs):
            special_symbols[i.value] = t.value

        # Generate objective for vocab of torchtext
        for ln in LANGUAGES: 
            vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(data_iter, ln),
                                                            min_freq=2,
                                                            specials=special_symbols.values(),
                                                            special_first=True)
        # Set 'unk' as default index. If token is not founded, return default index
        for ln in LANGUAGES: 
            vocab_transform[ln].set_default_index(IDs.UNK.value)

        
        for ln in LANGUAGES: 
            text_transform[ln] = sequential_transforms(token_transform[ln],  # Tokenization
                                                       vocab_transform[ln],  # Numericalization
                                                       tensor_transform)     # Add BOS/EOS and make tensor
        # input_dim, output_dim
        SRC_VOCAB_SIZE=len(vocab_transform[self.SRC_LANG])
        TGT_VOCAB_SIZE=len(vocab_transform[self.TGT_LANG])
        self.collate_fn = Collate_fn(text_transform[self.SRC_LANG], 
                                     text_transform[self.TGT_LANG])
        print("SRC_VOCAB_SIZE : %d" %(SRC_VOCAB_SIZE))
        print("TGT_VOCAB_SIZE : %d" %(TGT_VOCAB_SIZE))

    def train_dataloader(self):
        return DataLoader(self.train_iter, batch_size=self.batch_size, 
                          collate_fn=self.collate_fn, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.valid_iter, batch_size=self.batch_size, 
                          collate_fn=self.collate_fn, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_iter, batch_size=self.batch_size, 
                          collate_fn=self.collate_fn, num_workers=4)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch_size', type=int, default=16)
        return parser

class Collate_fn:
    def __init__(self, src_text_transform, tgt_text_transform):
        self.src_text_transform = src_text_transform
        self.tgt_text_transform = tgt_text_transform

    def __call__(self, batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_text = self.src_text_transform(src_sample.rstrip('\n'))
            tgt_text = self.tgt_text_transform(tgt_sample.rstrip('\n'))
            src_batch.append( src_text )
            tgt_batch.append( tgt_text )
        src_batch = pad_sequence(src_batch, padding_value=IDs.PAD.value)
        tgt_batch = pad_sequence(tgt_batch, padding_value=IDs.PAD.value)
        return src_batch, tgt_batch
