# -*- coding: UTF-8 -*-

from __future__ import absolute_import

import cv2
import numpy as np
from ..utils.processing import bgr_to_gray
from ..utils.processing import gray_to_bgr


class BinaryThresholding:
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, img):
        gray = bgr_to_gray(img)
        (_, binary_mask) = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        binary_mask = np.expand_dims(binary_mask, 2)
        connectivity = 4
        _, label_img, _, _ = cv2.connectedComponentsWithStats(binary_mask , connectivity , cv2.CV_32S)
        return label_img
    