# -*- coding: UTF-8 -*-

from __future__ import absolute_import

import cv2
import numpy as np


class GradCut:
    def __init__(self, iteration=3):
        self.iteration = iteration

    @staticmethod
    def bbox_verify(rect, img_sz):
        height, width = img_sz
        xmin = max(rect[0], 0)
        ymin = max(rect[1], 0)
        xmax = min(rect[2], width)
        ymax = min(rect[3], height)
        return (xmin, ymin, ymax - ymin, xmax - xmin)

    def __call__(self, img, rect, obj_idx):
        img_sz = img.shape[0:2]
        rect = self.bbox_verify(rect, img_sz)
        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        # Grabcut
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, self.iteration, cv2.GC_INIT_WITH_RECT)
        binary_mask = np.where((mask == 2) | (mask == 0), 0, obj_idx).astype('uint8')
        return np.expand_dims(binary_mask, axis=2)


def get_grad_label_img():
    pass


if __name__ == '__main__':
    def visual(img, label_img):
        label = np.unique(label_img)
        height, width = img.shape[:2]
        visual_img = np.zeros((height, width, 3))
        for lab in label:
            if lab == 0:
                continue
            color = np.random.randint(low=0, high=255, size=3)
            visual_img[label_img==lab, :] = color
        return img.astype(np.uint8), visual_img.astype(np.uint8)

    img = cv2.imread('../../../debug_scripts/test_imgs/min_max.jpg')
    seg = GradCut()
    label_img = seg(img, rect=[85,85,85+55,85+55], obj_idx=1)
    img, visual_img = visual(img, label_img)
    cv2.imwrite('./img.png', img)
    cv2.imwrite('./results.png', visual_img)
