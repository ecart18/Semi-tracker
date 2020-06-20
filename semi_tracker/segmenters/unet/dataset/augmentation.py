# -*- coding: UTF-8 -*-

from __future__ import print_function, absolute_import
import random
import math
import cv2
import numpy as np


class RandomFlip:
    def __init__(self, prob):
        self.prob = prob
    def __call__(self, img, label):
        if random.random() < self.prob:
            d = random.randint(-1, 1)
            img = cv2.flip(img, d)
            label = cv2.flip(label, d)
        return img, label


class RandomRotate:
    def __init__(self, prob):
        self.prob = prob
    def __call__(self, img, label):
        if random.random() < self.prob:
            factor = random.randint(0, 4)
            img = np.rot90(img, factor)
            label = np.rot90(label, factor)
        return img, label


class GaussainNoise:
    def __init__(self, prob):
        self.prob = prob
        self.mean = 0
        self.var = 0.1
	
    def __call__(self, img, label):
        if random.random() < self.prob:
            img = img.astype(np.uint8)
            h, w, c = img.shape
            sigma = self.var ** 0.5
            gauss = np.random.normal(self.mean, sigma, (h, w, c))
            gauss = gauss.reshape(h, w, c).astype(np.uint8)
            img = img + gauss
        return img, label


class GaussainBlur:
    def __init__(self, prob):
        self.prob = prob
        self.max_filter_size = 5
        self.sigma = 0.1 ** 0.5
	
    def __call__(self, img, label):
        if random.random() < self.prob:
            img = img.astype(np.uint8)
            filter_size = random.randint(3, self.max_filter_size)
            if filter_size % 2 == 0:
                filter_size += 1
            img = cv2.GaussianBlur(img, (filter_size, filter_size), self.sigma)
        return img, label


class Compose:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img, label):
        assert img.size == label.size
        for tf in self.transform:
            if tf:
                img, label = tf(img, label)
                assert img.size == label.size
        if not isinstance(img, np.ndarray):
            img, label = np.array(img), np.array(label, dtype=np.uint8)
        return img, label


__factory__ = {
    'Flip': RandomFlip,
    'Rotate': RandomRotate,
    'GaussainNoise': GaussainNoise,
    'GaussainBlur': GaussainBlur
}


class Augmentation(object):

    def __init__(self, aug_list=None):
        self.augment = self.gen_compose(aug_list)

    def gen_compose(self, aug_list):

        augments = []
        prob = 0.5 / len(aug_list)
        if aug_list:
            for aug in aug_list:
                tf = __factory__.get(aug, None)
                if tf:
                    augments.append(tf(prob=prob))
        return Compose(augments)