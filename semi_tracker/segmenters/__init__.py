from __future__ import absolute_import
from .customized.customized import CustomizedSegmentor
from .tradition_methods.threshold import BinaryThresholding
from .tradition_methods.watershed import WaterShed
from .tradition_methods.gradcut import GradCut
from .unet.unet import Unet

__all__ = [
    'BinaryThresholding',
    'WaterShed',
    'GradCut',
    'Unet',
    'CustomizedSegmentor',
    'get_segmenter'
]

__SEGMENTER = {
    'binary_thresholding': BinaryThresholding,
    'water_shed': WaterShed,
    'grad_cut': GradCut,
    'unet': Unet,
    'customized_segmentor': CustomizedSegmentor
}


def get_segmenter(name, **kwargs):
    """
    :name: segmenter name
    :return: segmenter
    """
    if name not in __SEGMENTER:
        raise KeyError("Unknown segmenter:", name)
    segmenter = __SEGMENTER[name](**kwargs)
    return segmenter
