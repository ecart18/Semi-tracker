# -*- coding: UTF-8 -*-

from __future__ import print_function, absolute_import
import cv2
import numpy as np


def find_edge(masks):
    edge = np.zeros_like(masks)
    unique_label = np.delete(np.unique(masks), 0)
    for label in unique_label:
        coords = np.where(masks == label)
        mask = np.zeros_like(masks)
        mask[coords[0], coords[1]] = 1
        temp_edge = cv2.Canny(mask.astype(np.uint8), 2, 5)
        temp_edge = cv2.dilate(temp_edge, kernel=np.ones(
            (3, 3), np.uint8), iterations=1)
        edge += temp_edge
    return edge


def make_balance_weight_map(masks):
    if masks.ndim == 2:
        masks = np.expand_dims(masks, 0)
    nrows, ncols = masks.shape[1:]
    masks = (masks > 0).astype(int)
    # class weight map
    w_map = np.zeros((nrows, ncols))
    w_0 = masks.sum()
    w_1 = w_map.size - w_0
    w_map[masks.sum(0) == 1] = w_1 / w_map.size
    w_map[masks.sum(0) == 0] = w_0 / w_map.size
    w_map = np.expand_dims(w_map, axis=0)
    return w_map.astype(np.float32)


def make_weight_map_instance(masks, w0=10, sigma=5):
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
            maps[:, :, idx] = cv2.distanceTransform(
                1 - (masks == label).astype(np.uint8), cv2.DIST_L2, 3)
            idx += 1
    maps = np.sort(maps, axis=2)
    d1 = maps[:, :, 0]
    d2 = maps[:, :, 1]
    dis = ((d1 + d2)**2) / (2 * sigma * sigma)
    dw_map = w0*np.exp(-dis) * (masks == 0)

    masks = 1 * (masks > 0)
    c_weights = np.zeros(2)
    clsw_map = np.zeros_like(masks)
    c_weights[1] = 1 - masks.sum() / masks.size
    c_weights[0] = 1 - c_weights[1]
    c_weights /= c_weights.max()
    clsw_map = np.where(masks == 0, c_weights[0], c_weights[1])
    w_map = clsw_map + dw_map
    w_map = np.expand_dims(w_map, axis=0)
    return w_map.astype(np.float32)


def make_weight_map_ins(masks, w0 = 10, sigma = 5):
    from skimage import measure
    edges = find_edge(masks)
    masks[edges > 0] = 0
    cells = measure.label(masks, connectivity=2)
    dw_map = np.zeros_like(masks)
    maps = np.zeros((masks.shape[0], masks.shape[1], cells.max()))
    if cells.max()>=2:
        for i in range(1, cells.max() + 1):
            tmp = 1- (cells == i).astype(np.uint8)
            maps[:,:,i-1] =  cv2.distanceTransform(1- (cells == i).astype(np.uint8), cv2.DIST_L2, 3)
    maps = np.sort(maps, axis = 2)
    d1 = maps[:,:,0]
    d2 = maps[:,:,1]
    dis = ((d1 + d2)**2) / (2 * sigma * sigma)
    dw_map = w0*np.exp(-dis) * (cells == 0)
    # dw_map = w0*np.exp(-dis)

    masks = 1 * (masks > 0)
    c_weights = np.zeros(2)
    clsw_map = np.zeros_like(masks)
    c_weights[1] = 1 - masks.sum() / masks.size
    c_weights[0] = 1 - c_weights[1]
    c_weights /= c_weights.max()
    clsw_map = np.where(masks==0, c_weights[0], c_weights[1])
    w_map = clsw_map + dw_map

    # w_map = cv2.resize(w_map, (1024, 356))
    w_map = np.expand_dims(w_map, axis=0)
    return w_map.astype(np.float32)


def label_reader(label_path):
    def label_standardization(img):
        if len(img.shape) == 3:
            img = np.squeeze(img, 2)
            return img
        elif len(img.shape) == 2:
            return img
        else:
            raise TypeError(
                'The label image shape dimension is not equal to 2 or 3.')

    try:
        extentions = label_path.split('.')[-1].lower()
        if extentions in ['png', 'tif']:
            label = cv2.imread(label_path, -1)
            label = label_standardization(label)
            return label
        else:
            raise TypeError(
                'The label image type of {} is not supported in training yet.'.format(extentions))
    except:
        raise ValueError('Load label image {} failed.'.format(label_path))

if __name__ == "__main__":
    label_image_path = '../../../../training_demo2/label_img/annotation_000000.png'
    label_image = label_reader(label_image_path)
    weight1 = make_weight_map_instance(label_image)
    weight1 = 255 * (weight1 / weight1.max())
    weight1 = weight1.astype(np.uint8)
    cv2.imwrite('./weight1.jpg', np.squeeze(weight1, 0))
    
    label_image_path = '../../../../training_demo2/label_img/annotation_000000.png'
    label_image = label_reader(label_image_path)
    weight2 = make_weight_map_ins(label_image)
    weight2 = 255 * (weight2 / weight2.max())
    weight2 = weight2.astype(np.uint8)
    cv2.imwrite('./weight2.jpg', np.squeeze(weight2, 0))
    import pdb
    pdb.set_trace()
