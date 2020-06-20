# -*- coding: UTF-8 -*-

from __future__ import print_function, absolute_import

import sys
import random
import os.path as osp
import numpy as np
import torch
from torch import nn
from torch.backends import cudnn

# CURRENT_DIR = osp.dirname(__file__)
# sys.path.append(CURRENT_DIR)
# sys.path.append(osp.join(CURRENT_DIR, '..'))

from .backbone import get_backbone
from .trainer import get_trainer
from .loss import build_loss
from .dataset import build_dataloader
from .utils import Logger, load_checkpoint, save_checkpoint, export_history


def train(TrainParameters):

    random.seed(TrainParameters.train_seed)
    np.random.seed(TrainParameters.train_seed)
    torch.manual_seed(TrainParameters.train_seed)
    cudnn.benchmark = True

    # Redirect print to both console and log file
    sys.stdout = Logger(osp.join(TrainParameters.log_root, 'train_log.txt'))

    train_loader, val_loader = build_dataloader(**TrainParameters.dataloader_params)
    model = get_backbone(name='unet').to(TrainParameters.device)
    if TrainParameters.gpu_num > 1:
        model = nn.DataParallel(model).to(TrainParameters.device)
    else:
        model = model.to(TrainParameters.device)

    criterion = build_loss(name=TrainParameters.loss_type).to(TrainParameters.device)

    optimizer = torch.optim.Adam(model.parameters(), **TrainParameters.optimizer_params)

    # Load from checkpoint
    start_epoch = 0
    best_loss = 1e8
    header = ['epoch', 'train_loss', 'val_loss']

    # Trainer
    trainer = get_trainer(name='unet_trainer', model=model, criterion=criterion)

    if TrainParameters.resume:
        print("load previous checkpoint file from {} \n".format(TrainParameters.resume))
        checkpoint = load_checkpoint(TrainParameters.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        print("=> Start epoch {}  best_loss {:.3%}"
              .format(start_epoch, best_loss))

    # Start training
    for epoch in range(start_epoch, TrainParameters.epochs):

        train_loss = trainer.train(epoch, train_loader, optimizer, device=TrainParameters.device)
        val_loss = trainer.eval(epoch, val_loader, device=TrainParameters.device)

        values = [epoch + 1, train_loss, val_loss]
        export_history(header=header, value=values, file_path=osp.join(TrainParameters.log_root, "loss_per_epoch.csv"))

        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        save_checkpoint({
            'state_dict': model.module.state_dict(),
            'epoch': epoch + 1,
            'best_loss': best_loss,
        }, is_best, fpath=osp.join(TrainParameters.log_root, 'best_checkpoint.pth.tar'))

        print('\n * Finished epoch {:3d}  train_loss: {:.3f}  val_loss: {:.3f}  best: {:.3f}{}\n'.
              format(epoch, train_loss, val_loss, best_loss, ' *' if is_best else ''))

