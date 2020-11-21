# -*- coding: UTF-8 -*-

from __future__ import absolute_import

import cv2
import numpy as np


class EqualizeHist:
    def __init__(self):
        pass

    def __call__(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        # equalize the histogram of the Y channel
        img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
        # convert the YUV image back to RGB format
        img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
        return img


# Contrast limited adaptive histogram equalization
class CLAHE:
    def __init__(self, clip_limit=2.0, grid_size=8):
        self.clahe = cv2.createCLAHE(clipLimit=3.0, 
                                    tileGridSize=(int(grid_size), int(grid_size))) 
    
    def __call__(self, img):
        # Converting image to LAB Color model
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        # Splitting the LAB image to different channels
        l, a, b = cv2.split(img)
        # Applying CLAHE to L-channel
        cl = self.clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        # Converting image from LAB Color model to RGB model
        img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return img