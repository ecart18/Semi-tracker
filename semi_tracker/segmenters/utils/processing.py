# -*- coding: UTF-8 -*-

from __future__ import absolute_import

import cv2
import numpy as np

def bgr_to_gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def gray_to_bgr(img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img