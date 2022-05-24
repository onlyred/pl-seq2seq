import argparse
import shutil
import traceback
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from models import Seq2Seq
from data import Multi30k_DataModule

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='0', help='number of gpus')
    parser.add_argument('--seed', type=int, default=0, help='seed number')
    parser.add_argument('--logdir', type=str, default='./logs', help='log directory')
    parser.add_argument('--epochs', type=int, default=1000, help='max_epochs')
    parser.add_argument('--mode', choices=['train','test'], type=str, help='train/test')
    parser.add_argument('--saved_model', type=str, default='tut1-model.pt', help='saved model')
    return parser

def main():
    parser = get_arguments()
    hparams, _ = parser.parse_known_args()
    pl.seed_everything(hparams.seed)

    modelname = 'seq2seq'
    Model = Seq2Seq
    parser = Model.add_model_specific_args(parser)
    hparams = parser.parse_args()

    DataModule = Multi30k_DataModule
    parser = DataModule.add_model_specific_args(parser)
    hparams = parser.parse_args()

    logger = TensorBoardLogger(hparams.logdir, name=modelname)
    logger.log_hyperparams(dict(hparams.__dict__))
    logpath= Path("%s/%s/version_%s" %(hparams.logdir, modelname, logger.version))
    logpath.mkdir(parents=True, exist_ok=True)

    dm = DataModule(hparams)
    model = Model(hparams)
    # define callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath = logpath,
        monitor='val_loss',
        save_weights_only=True,
        mode="min",
    )
    ealry_stop_callback = EarlyStopping(
        patience=3,
        min_delta=0.00,
        monitor='val_loss',
        verbose=False,
        mode="min",
    )
    # define trainer
    trainer = pl.Trainer(
        callbacks=[ealry_stop_callback, checkpoint_callback],
        logger= logger,
        gpus= [int(i) for i in hparams.gpus.split(',')],
        max_epochs=hparams.epochs,
    )
    # start training
    ierror=False
    try:
        if hparams.mode == 'train':
            trainer.fit(model, datamodule=dm)
        else:
            model.load_from_checkpoint(hparams.saved_model)
            trainer.test(model, datamodule=dm)
    except KeyboardInterrupt as e:
        traceback.print_stack()
        traceback.print_exc()
        ierror=True
    finally:
        if ierror:
            shutil.rmtree(logpath)
            print('Stop Running : Remove  %s' %(logpath))

if __name__ == "__main__":
    main()
