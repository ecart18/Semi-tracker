# -*- coding: UTF-8 -*-

from __future__ import print_function, absolute_import

import time
import torch
from ..utils.meters import AverageMeter


class BaseTrainer(object):
    def __init__(self, model, criterion):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion

    def train(self, epoch, data_loader, optimizer, device=torch.device('cuda'), print_freq=1):

        self.model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            loss, batch_size = self._forward(inputs, device)

            losses.update(loss.item(), batch_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('train: \t'
                      'Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.5f} ({:.5f})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))
        return losses.avg

    def eval(self, epoch, data_loader, device=torch.device('cuda'), print_freq=1):

        self.model.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        end = time.time()
        with torch.no_grad():
            for i, inputs in enumerate(data_loader):
                data_time.update(time.time() - end)

                loss, batch_size = self._forward(inputs, device)

                losses.update(loss.item(), batch_size)

                batch_time.update(time.time() - end)
                end = time.time()

                if 0 == (i + 1) % print_freq:
                    print('validation: \t'
                          'Epoch: [{}][{}/{}]\t'
                          'Time {:.3f} ({:.3f})\t'
                          'Data {:.3f} ({:.3f})\t'
                          'Loss {:.5f} ({:.5f})\t'
                          .format(epoch, i + 1, len(data_loader),
                                  batch_time.val, batch_time.avg,
                                  data_time.val, data_time.avg,
                                  losses.val, losses.avg))

        return losses.avg


    def _forward(self, inputs, device):
        raise NotImplementedError


class Trainer(BaseTrainer):

    def _parse_data(self, inputs, device):
        data, gt = inputs["image"], inputs["gt"]
        data, gt = data.to(device), gt.to(device)
        weight = None
        if inputs["image"] is not None:
            weight = inputs["weight"]
            weight = weight.to(device)
        return {"data": data,
                "gt": gt,
                "weight": weight}

    def _forward(self, inputs, device):
        inputs = self._parse_data(inputs, device)
        batch_size = inputs["gt"].size(0)
        predictions = self.model(inputs["data"])
        loss = 10.*self.criterion(predictions, inputs["gt"], inputs["weight"])
        # loss = self.criterion(predictions, inputs["gt"])
        return loss, batch_size
