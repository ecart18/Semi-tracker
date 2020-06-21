# -*- coding: UTF-8 -*-

from __future__ import print_function, absolute_import

import sys
import random
import os.path as osp
import numpy as np
import torch
from torch import nn
from torch.backends import cudnn

from .backbone import get_backbone
from .trainer import get_trainer
from .loss import build_loss
from .dataset import build_dataloader
from .utils import Logger, load_checkpoint, save_checkpoint, export_history


def train(train_parameters):

    random.seed(train_parameters.train_seed)
    np.random.seed(train_parameters.train_seed)
    torch.manual_seed(train_parameters.train_seed)
    cudnn.benchmark = True

    # Redirect print to both console and log file
    sys.stdout = Logger(osp.join(train_parameters.log_root, 'train_log.txt'))

    train_loader, val_loader = build_dataloader(name='cells', **train_parameters.dataloader_params)
    model = get_backbone(name='unet').to(train_parameters.device)
    if train_parameters.gpu_num > 1:
        model = nn.DataParallel(model).to(train_parameters.device)
    else:
        model = model.to(train_parameters.device)

    criterion = build_loss(name=train_parameters.loss_type).to(train_parameters.device)

    optimizer = torch.optim.Adam(model.parameters(), **train_parameters.optimizer_params)

    # Load from checkpoint
    start_epoch = 0
    best_loss = 1e8
    header = ['epoch', 'train_loss', 'val_loss']

    # Trainer
    trainer = get_trainer(name='unet_trainer', model=model, criterion=criterion)

    if train_parameters.resume:
        try:
            print("load previous checkpoint file from {} \n".format(train_parameters.resume))
            checkpoint = load_checkpoint(train_parameters.resume)
            model.load_state_dict(checkpoint['state_dict'])
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            print("=> Start epoch {}  best_loss {:.3%}"
                .format(start_epoch, best_loss))
        except:
            raise ValueError('Load previous checkpoint file {} failed.".format(train_parameters.resume)')

    # Start training
    for epoch in range(start_epoch, train_parameters.epochs):

        train_loss = trainer.train(epoch, train_loader, optimizer, device=train_parameters.device)
        val_loss = trainer.eval(epoch, val_loader, device=train_parameters.device)

        values = [epoch + 1, train_loss, val_loss]
        export_history(header=header, value=values, file_path=osp.join(train_parameters.log_root, "loss_per_epoch.csv"))

        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        save_checkpoint({
            'state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'best_loss': best_loss,
        }, is_best, fpath=osp.join(train_parameters.log_root, 'checkpoint.pth.tar'))

        print('\n * Finished epoch {:3d}  train_loss: {:.3f}  val_loss: {:.3f}  best: {:.3f}{}\n'.
              format(epoch, train_loss, val_loss, best_loss, ' *' if is_best else ''))

