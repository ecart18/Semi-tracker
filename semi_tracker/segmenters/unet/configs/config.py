# -*- coding: UTF-8 -*-

from __future__ import absolute_import
import torch
import os.path as osp
from ..utils import mkdir



class TrainParameters:

    def __init__(self, source_img_root, label_img_root, log_root, 
                    validation_ratio, sample_balance, edge_weighted,
                    batch_size, workers, gpu_num, resume, epochs,
                    lr, weight_decay, loss_type, **kwargs):

        # train
        self.train_seed = 0        
        self.resume = resume
        self.epochs, self.gpu_num, self.device = self._check_train(epochs, gpu_num)
        
        #IO
        self.source_img_root, self.label_img_root, self.log_root = self._check_io(source_img_root, label_img_root, log_root)

        # dataloader
        if sample_balance is None:
            sample_balance = True
        if edge_weighted is None:
            edge_weighted = False
        batch_size, validation_ratio, workers= self._check_dataloader(batch_size, validation_ratio, workers)

        self.dataloader_params = {
            'source_img_root': self.source_img_root,
            'label_img_root': self.label_img_root,
            'log_root': self.log_root,
            'validation_ratio': validation_ratio,
            'sample_balance': sample_balance,
            'edge_weighted': edge_weighted,
            'batch_size': batch_size,
            'workers': workers
        }

        # optimizer
        self.lr, self.weight_decay = self._check_optimizer(lr, weight_decay)
        self.optimizer_params = {
            'weight_decay' : self.weight_decay,
            'lr' : self.lr
        }

        # loss
        self.loss_type = self._check_loss(loss_type)

       
    
    def _check_dataloader(self, batch_size, validation_ratio, workers):
        if batch_size:
            try:
                assert isinstance(batch_size, int)
            except AssertionError:
                raise TypeError('batch size should be int type.')
        else:
            batch_size = 2

        if workers:
            try:
                assert isinstance(workers, int)
            except AssertionError:
                raise TypeError('workers should be int type')
        else:
            workers = 2

        if validation_ratio:
            try:
                assert isinstance(validation_ratio, float)
            except AssertionError:
                raise TypeError('validation ratio should be float type')
        else:
            validation_ratio = 0.2

        return batch_size, validation_ratio, workers


    def _check_train(self, epochs, gpu_num):
        if epochs:
            try:
                assert isinstance(epochs, int)
            except AssertionError:
                raise TypeError('epochs should be int type.')
        else:
            epochs = 100
        
        if gpu_num:
            try:
                assert isinstance(gpu_num, int)
            except AssertionError:
                raise TypeError('gpu number should be int type')
            device = 'cpu'
            if gpu_num > 0 and torch.cuda.is_available():
                device = 'cuda'
            else:
                gpu_num = 0
                device = 'cpu'
        else:
            gpu_num = 0
            device = 'cpu'

        return epochs, gpu_num, device


    def _check_optimizer(self, lr, weight_decay):
        if lr and weight_decay:
            try:
                assert isinstance(lr, float)
            except AssertionError:
                raise TypeError('Learning rate should be float type.')
            try:
                assert isinstance(lr, float)
            except AssertionError:
                raise TypeError('Learning rate should be float type.')
        else:
            lr = 0.001
            weight_decay = 0.0005
        return lr, weight_decay
        

    def _check_loss(self, loss_type):
        if loss_type:
            try:
                assert isinstance(loss_type, str)
            except AssertionError:
                raise TypeError('Loss type should be string type.')
            return loss_type
        else:
            loss_type = 'WeightedSoftDiceLoss'

        return loss_type
            

    def _check_io(self, source_img_root, label_img_root, log_root):
        try:
            assert osp.exists(source_img_root)
        except AssertionError:
            raise ValueError('Path {} is not exist.'.format(source_img_root))
        try:
            assert osp.exists(label_img_root)
        except AssertionError:
            raise ValueError('Path {} is not exist.'.format(label_img_root))
        mkdir(log_root)
        return source_img_root, label_img_root, log_root
        


