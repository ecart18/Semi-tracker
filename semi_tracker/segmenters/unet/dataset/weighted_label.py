# -*- coding: UTF-8 -*-

from __future__ import print_function, absolute_import
import cv2
import numpy as np


def find_edge(masks):
    edge = np.zeros_like(masks).astype(np.float32)
    unique_label = np.delete(np.unique(masks), 0)
    for label in unique_label:
        coords = np.where(masks=label)
        mask = np.zeros_like(masks)
        mask[coords[:, 0], coords[:, 1]] = 1
        temp_edge = cv2.Canny(mask.astype(np.uint8), 2, 5) / 255
        temp_edge = cv2.dilate(img,kernel=3,iterations = 1)
        edge += temp_edge
    return np.clip(edge, a_min=0, a_max=1).astype(np.float32)


def make_balance_weight_map(masks):
    if masks.ndim == 2:
        masks = np.expand_dims(masks, 0)
    nrows, ncols = masks.shape[1:]
    masks = (masks > 0).astype(int)
    # class weight map
    w_map = np.zeros((nrows, ncols))
    w_1 = w_map.size - masks.sum()
    w_0 = masks.sum()
    w_map[masks.sum(0) == 1] = w_1
    w_map[masks.sum(0) == 0] = 1.0*w_0
    w_map = np.expand_dims(w_map, axis=0)
    return w_map.astype(np.float32)


def make_weight_map_instance(masks, w0 = 10, sigma = 5):
    """
    Generate the weight maps as specified in the UNet paper
    Parameters
    ----------
    masks: array-like
        A 2D label array of shape image_height, image_width),
    w0: scalar
    sigma: scalar

    Returns
    -------
    array-like
        A 3D array of shape (1, image_height, image_width)

    """
    edges = find_edge(masks)
    masks[edges > 0] = 0
    unique_label = np.delete(np.unique(masks), 0)
    dw_map = np.zeros_like(masks)
    maps = np.zeros((masks.shape[0], masks.shape[1], len(unique_label)))
    idx = 0
    if len(unique_label) >= 1:
        for label in unique_label:
            maps[:,:,idx] =  cv2.distanceTransform(1- (masks == label).astype(np.uint8), cv2.DIST_L2, 3)
            idx += 1
    maps = np.sort(maps, axis = 2)
    d1 = maps[:,:,0]
    d2 = maps[:,:,1]
    dis = ((d1 + d2)**2) / (2 * sigma * sigma)
    dw_map = w0*np.exp(-dis) * (masks == 0)

    masks = 1 * (masks > 0)
    c_weights = np.zeros(2)
    clsw_map = np.zeros_like(masks)
    c_weights[1] = 1 - masks.sum() / masks.size
    c_weights[0] = 1 - c_weights[1]
    c_weights /= c_weights.max()
    clsw_map = np.where(masks==0, c_weights[0], c_weights[1])
    w_map = clsw_map + dw_map
    w_map = np.expand_dims(w_map, axis=0)
    w_map = cv2.resize(w_map, (1024, 356))
    return w_map.astype(np.float32)


if __name__ == "__main__":
    # gt_path = "/home/taohu/celltracking/KalmanSeg/data/DIC-C2DH-HeLa/images/01_GT/SEG/man_seg002.tif"
    # gt = cv2.imread(gt_path, -1)
    # if len(gt.shape) == 3:
    #     gt = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
    # # if gt.max() == 1:
    #     # gt = gt * 255
    # # gt[gt > 0] = 255
    # gt = gt.astype(np.uint8)
    # cv2.imwrite('gt002.jpg', gt*255)
    # weight = make_weight_map_ins(gt, w0=10, sigma=5)
    # # weight = (weight * (255/ weight.max())).astype(np.uint8)
    # # cv2.imwrite('weight002.jpg', (weight.squeeze(0)* (255/ weight.max())).astype(np.uint8))
    # cv2.imwrite('weight002.jpg', weight * (255/ weight.max()))
    # print(weight.min())
    # print(weight.max())
    # import matplotlib.pyplot as plt
    # plt.figure(figsize = (10,10))
    # plt.imshow(weight, cmap = 'jet')
    # plt.colorbar()
    # plt.savefig('./weight_color002.jpg')

    import cv2
    gt_path = "/home/taohu/celltracking/KalmanSeg/data/Ecoli-Exp/images/gt/Project.lif_Series002_Crop005_t00.png"
    gt = cv2.imread(gt_path, -1)
    if len(gt.shape) == 3:
        gt = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
    gt = gt.astype(np.uint8)
    edges = find_edge(gt)
    gt[edges > 0] = 0
    if gt.max() == 1:
        gt = gt * 255
    gt[gt > 0] = 255
    gt = gt.astype(np.uint8)

    gt_tmp = np.copy(gt)
    gt_tmp[gt_tmp >0] = 255
    cv2.imwrite('gt002.jpg', gt_tmp)
    weight = make_weight_map_ins(gt, w0=10, sigma=5)
    # weight = (weight * (255/ weight.max())).astype(np.uint8)
    # cv2.imwrite('weight002.jpg', (weight.squeeze(0)* (255/ weight.max())).astype(np.uint8))
    cv2.imwrite('weight002.jpg', weight * (255/ weight.max()))
    print(weight.min())
    print(weight.max())
    import matplotlib.pyplot as plt
    plt.figure(figsize = (10,10))
    plt.imshow(weight, cmap = 'jet')
    plt.colorbar()
    plt.savefig('./weight_color002.jpg')
