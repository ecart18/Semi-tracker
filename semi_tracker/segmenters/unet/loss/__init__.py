# -*- coding: UTF-8 -*-

from __future__ import absolute_import

from .loss import (DiceLoss, WeightedSoftDiceLoss, BCELoss, WBCELoss, MSELoss, MAELoss)

__all__ = [
    'DiceLoss',
    'WeightedSoftDiceLoss',
    'BCELoss',
    'WBCELoss',
    'MSELoss',
    'MAELoss',
    'bulid_loss'
]


__LOSS = {
    'DiceLoss': DiceLoss,
    'WDiceLoss': WeightedSoftDiceLoss,
    'BCELoss': BCELoss,
    'WBCELoss': WBCELoss,
    'MSELoss': MSELoss,
    'MAELoss': MAELoss
}


def build_loss(name, *args, **kwargs):
    if name not in __LOSS:
        raise KeyError("Unknown loss:", name)
    return __LOSS[name](*args, **kwargs)



