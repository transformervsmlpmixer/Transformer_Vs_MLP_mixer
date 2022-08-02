import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import comet_ml
from comet_ml import Experiment
import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger

from comet_ml import get_config
from datetime import datetime

import argparse

import comet_ml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np
import dill
from pathlib import Path
from utils import results_saver
from utils import results_dir_path, get_num_of_params
from config import Config
from utils import get_model, get_dataset, get_experiment_name, get_criterion

import warnings
from utils import results_cleaner

warnings.filterwarnings('ignore')
warnings.simplefilter("ignore")
config = Config()

torch.manual_seed(config.seed)
np.random.seed(config.seed)
config.benchmark = False
config.gpus = 1  # torch.cuda.device_count()
config.num_workers = 4 * config.gpus if config.gpus else 8
config.off_cls_token = False
config.is_cls_token = True if not config.off_cls_token else False

################# MY CODE ###############
num_epoch_cnt = 1
num_epoch_val_cnt = 0
num_epoch_train_cnt = 0

sum_acc, sum_loss = .0, .0
sum_val_loss, sum_val_acc = .0, .0

avg_loss_dct, avg_acc_dct = dict(), dict()
avg_val_loss_dct, avg_val_acc_dct = dict(), dict()

results_dir_path = results_dir_path()
#########################################

if not config.gpus:
    config.precision = 32

train_ds, test_ds = get_dataset(config)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                                       num_workers=config.num_workers,
                                       pin_memory=True)

test_dl = torch.utils.data.DataLoader(test_ds, batch_size=config.eval_batch_size, num_workers=config.num_workers,
                                      pin_memory=True)


class Net(pl.LightningModule):
    def __init__(self, hparams):
        super(Net, self).__init__()
        # self.hparams = hparams
        self.hparams.update(vars(hparams))
        self.model = get_model(hparams)
        self.criterion = get_criterion(config)
        self.log_image_flag = True
        ##################

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr,
                                          betas=(self.hparams.beta1, self.hparams.beta2),
                                          weight_decay=self.hparams.weight_decay)
        return [self.optimizer]

    def training_step(self, batch, batch_idx):
        # print(f'Training step - {batch_idx}')
        img, label = batch

        out = self.model(img)
        loss = self.criterion(out, label)
        acc = torch.eq(out.argmax(-1), label).float().mean()

        # print(f'label {label}')
        # print(f'img_shape: {img.shape}')
        # print(f'label.shape: {label.shape}')
        # print(f'out.shape: {out.shape}')
        # print(f'loss: {loss}')
        # print(f'acc: {acc}')

        self.log("loss", loss)
        self.log("acc", acc)
        # my code
        global sum_acc, sum_loss, num_epoch_train_cnt
        sum_loss += loss.item()
        sum_acc += acc.item()
        num_epoch_train_cnt += 1
        #########
        return loss

    def training_epoch_end(self, outputs):
        # my code
        global num_epoch_cnt, num_epoch_val_cnt, num_epoch_train_cnt, sum_loss, sum_acc, sum_val_acc, sum_val_loss
        global avg_acc_dct, avg_loss_dct, avg_val_loss_dct, avg_val_acc_dct

        if num_epoch_train_cnt > 0:
            avg_loss_dct[num_epoch_cnt] = sum_loss / num_epoch_train_cnt
            avg_acc_dct[num_epoch_cnt] = sum_acc / num_epoch_train_cnt

        if num_epoch_val_cnt:
            avg_val_loss_dct[num_epoch_cnt] = sum_val_loss / num_epoch_val_cnt
            avg_val_acc_dct[num_epoch_cnt] = sum_val_acc / num_epoch_val_cnt

        num_epoch_cnt += 1
        num_epoch_val_cnt = num_epoch_train_cnt = 0
        sum_val_acc = sum_val_loss = sum_loss = sum_acc = 0

        results_saver(avg_loss_dct, avg_acc_dct, avg_val_loss_dct, avg_val_acc_dct)  # save dicts as pickles
        #########

    def validation_step(self, batch, batch_idx):
        img, label = batch
        out = self.model(img)
        loss = self.criterion(out, label)
        acc = torch.eq(out.argmax(-1), label).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        # my code
        global sum_val_acc, sum_val_loss, num_epoch_val_cnt
        sum_val_acc += acc.item()
        sum_val_loss += loss.item()
        num_epoch_val_cnt += 1
        #########
        return loss

    def _log_image(self, image):
        grid = torchvision.utils.make_grid(image, nrow=4)
        self.logger.experiment.log_image(grid.permute(1, 2, 0))

        print("[INFO] LOG IMAGE!!!")


now = datetime.now()
time_msg = f'{now.hour:.2f}_{now.minute:.2f}__'

if __name__ == "__main__":
    print(f'is cuda available {torch.cuda.is_available()}')
    net = Net(config)
    print(f'config: {config}')
    experiment_name = get_experiment_name(config)
    print(experiment_name)
    with_embed, without_embed = get_num_of_params(model=net)
    better_attention_msg = f'depth: {config.depth}, width: {config.width}, ratio: {config.depth / np.log2(config.width):.2f},' \
                           f' parameters - with-embed: {without_embed}, without-embed: {without_embed}'

    comet_logger = pl.loggers.CometLogger(
        api_key="Lhc8PEQaw0MJTa5nqxv4H5OKr",
        save_dir="logs",
        project_name="betterattention",
        experiment_name=time_msg + better_attention_msg + experiment_name
    )

    # if config.api_key:
    print("[INFO] Log with Comet.ml!")
    # refresh_rate = 0
    # else:
    print("[INFO] Log with CSV")
    csv_logger = pl.loggers.CSVLogger(
        save_dir="logs",
        name=experiment_name
    )
    refresh_rate = 1

    trainer = pl.Trainer(precision=config.precision, fast_dev_run=config.dry_run, gpus=config.gpus,
                         benchmark=config.benchmark, logger=[comet_logger, csv_logger],
                         max_epochs=config.max_epochs, weights_summary="full",
                         progress_bar_refresh_rate=refresh_rate)
    trainer.fit(model=net, train_dataloaders=train_dl, val_dataloaders=test_dl)

    model_path = f"weights/{experiment_name}.pth"
    if not config.dry_run:
        torch.save(net.state_dict(), model_path)
        # if args.api_key:
    # comet_logger.experiment.log_asset(file_name=experiment_name, file_data=model_path)
