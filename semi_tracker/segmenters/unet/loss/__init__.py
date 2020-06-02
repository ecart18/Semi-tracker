from __future__ import absolute_import

from .loss import (DiceLoss, RMSELoss, BCELoss, WBCELoss, 
        CELoss, WCELoss, MSELoss, MAELoss, WeightedSoftDiceLoss)

__all__ = [
    'DiceLoss',
    'WeightedSoftDiceLoss',
    'RMSELoss',
    'BCELoss',
    'WBCELoss',
    'CELoss',
    'WCELoss',
    'MSELoss',
    'MAELoss',
    'bulid_loss'
]


__LOSS = {
    'DiceLoss': DiceLoss,
    'WDiceLoss': WeightedSoftDiceLoss,
    'RMSELoss': RMSELoss,
    'CELoss': CELoss,
    'WBCELoss': WBCELoss,
    'BCELoss': BCELoss,
    'WCELoss': WCELoss,
    'MSELoss': MSELoss,
    'MAELoss': MAELoss
}


def bulid_loss(name, *args, **kwargs):
    if name not in __LOSS:
        raise KeyError("Unknown loss:", name)
    return __LOSS[name](*args, **kwargs)



