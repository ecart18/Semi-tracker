# -*- coding: UTF-8 -*-

from __future__ import absolute_import

from collections import OrderedDict
import os.path as osp
import cv2
from jinja2 import Environment, FileSystemLoader 
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .utils import mkdir


class VisualFrame():
    def __init__(self, label_txt, proj_name, cell_type, fps=None,):
        pass

    def _add_frame_index(self):
        pass

    def _add_bbox(self):
        pass

    def _add_edge(self):
        pass
    
    def _trajectory(self):
        pass

    



class VisualVideo():

    def __init__(self, ):
        pass
    
    def to_video(self,):
        pass

    def __call__(self,):
        pass


