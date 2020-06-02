# -*- coding: UTF-8 -*-

from __future__ import absolute_import

import cv2
import numpy as np
from ..components.frame import Frame


def unit16b2uint8(img):
    if img.dtype == 'uint8':
        return img
    elif img.dtype == 'uint16':
        img = img.astype(np.float32) / 65535.0 * 255.0
        return img.astype(np.uint8)
    else:
        raise TypeError('No such of img transfer type: {} for img'.format(img.dtype))

def img_standardization(img):
    # img = unit16b2uint8(img)
    img = cv2.convertScaleAbs(img)
    if len(img.shape) == 2:
        img = np.expand_dims(img, 2)
        img = np.tile(img, (1, 1, 3))
        return img
    elif len(img.shape) == 3:
        return img
    else:
        raise TypeError('The Depth of image large than 3 \n')


def load_images(file_names):
    extentions = file_names[0].split('.')[-1].lower()
    frames = {}
    if extentions in ['png', 'jpg', 'jpeg']:
        for idx, file_name in enumerate(file_names):
            img = cv2.imread(file_name)
            img = img_standardization(img)
            frame = Frame(file_name=file_name, 
                            frame_id=idx, 
                            raw_img=img)
            frames[idx] = frame

     # TODO       
    if extentions in ['mp4', 'avi', 'tif']:
        raise TypeError('No supported images format with extentions: {}.'.format(extentions))
    
    if not len(frames) >= 1:
        raise ValueError('load images failure \n')
    return frames