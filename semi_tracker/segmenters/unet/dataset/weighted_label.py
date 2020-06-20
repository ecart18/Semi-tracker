# -*- coding: UTF-8 -*-

from __future__ import print_function, absolute_import
import cv2
import numpy as np


def find_edge(masks):
    edge = np.zeros_like(masks)
    unique_label = np.delete(np.unique(masks), 0)
    for label in unique_label:
        coords = np.where(masks==label)
        mask = np.zeros_like(masks)
        mask[coords[0], coords[1]] = 1
        temp_edge = cv2.Canny(mask.astype(np.uint8), 2, 5)
        temp_edge = cv2.dilate(temp_edge, kernel=np.ones((3, 3), np.uint8), iterations = 1)
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


def make_weight_map_instance(masks, w0 = 10, sigma = 20):
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
    return w_map.astype(np.float32)


if __name__ == "__main__":

    def _label_reader(label_path):
        def label_standardization(img):
            if len(img.shape) == 3:
                img = np.squeeze(img, 2)
                return img
            elif len(img.shape) == 2:
                return img
            else:
                raise TypeError('The label image shape dimension is not equal to 2 or 3.')

        extentions = label_path.split('.')[-1].lower()
        if extentions in ['png', 'tif']:
            label = cv2.imread(label_path, -1)
            label = label_standardization(label)
            return label
        else:
            raise TypeError('The label image type of {} is not supported in training yet.'.format(extentions))

    import cv2
    label_path = "/Users/hutaobetter/Projects/semi-tracker/debug_scripts/test_imgs/DIC/man_seg002.tif"
    label = _label_reader(label_path)
    weight = make_weight_map_instance(label, w0=10, sigma=20)
    weight = np.squeeze(weight)
    cv2.imwrite('/Users/hutaobetter/Projects/semi-tracker/debug_scripts/test_imgs/DIC/weight_seg002.jpg', weight * (255 / weight.max()))
    print(weight.min())
    print(weight.max())
    import matplotlib.pyplot as plt
    plt.figure(figsize = (10,10))
    plt.imshow(weight, cmap = 'jet')
    plt.colorbar()
    plt.savefig('/Users/hutaobetter/Projects/semi-tracker/debug_scripts/test_imgs/DIC/weight_color_seg002.png')


    import cv2
    label_path = "/Users/hutaobetter/Projects/semi-tracker/debug_scripts/test_imgs/SIM/man_seg000.tif"
    label = _label_reader(label_path)
    weight = make_weight_map_instance(label, w0=10, sigma=20)
    # weight = make_balance_weight_map(label)
    weight = np.squeeze(weight)
    cv2.imwrite('/Users/hutaobetter/Projects/semi-tracker/debug_scripts/test_imgs/SIM/weight_seg000.jpg', weight * (255 / weight.max()))
    print(weight.min())
    print(weight.max())
    import matplotlib.pyplot as plt
    plt.figure(figsize = (10,10))
    plt.imshow(weight, cmap = 'jet')
    plt.colorbar()
    plt.savefig('/Users/hutaobetter/Projects/semi-tracker/debug_scripts/test_imgs/SIM/weight_color_seg000.png')
