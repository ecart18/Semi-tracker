# -*- coding: UTF-8 -*-

from __future__ import absolute_import

import os.path as osp
import cv2
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from .cell import create
from .augmentation import Augmentation
from .weighted_label import make_balance_weight_map, make_weight_map_instance


class Preprocessor(Augmentation):
    def __init__(self, dataset, mode, source_img_root, label_img_root, scale_img, weighted_type, aug_list):
        super(Preprocessor, self).__init__(aug_list=aug_list)
        self.dataset = dataset
        self.mode = mode
        self.scale_img = scale_img
        self.source_img_root = source_img_root
        self.label_img_root = label_img_root
        self.weighted_type = weighted_type

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _img_reader(self, image_path):
        def img_standardization(img):
            img = cv2.convertScaleAbs(img)
            if len(img.shape) == 2:
                img = np.expand_dims(img, 2)
                img = np.tile(img, (1, 1, 3))
                return img
            elif len(img.shape) == 3:
                return img
            else:
                raise TypeError('The Depth of image large than 3.')
        
        try:
            extentions = image_path.split('.')[-1].lower()
            if extentions in ['png', 'jpg', 'jpeg', 'tif']:
                img = cv2.imread(image_path, -1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img_standardization(img)
                return img
            else:
                raise TypeError('The image type of {} is not supported in training yet.'.format(extentions))
        except:
            raise ValueError('Load image {} failed.'.format(image_path))


    def _label_reader(self, label_path):
        def label_standardization(img):
            if len(img.shape) == 3:
                img = np.squeeze(img, 2)
                return img
            elif len(img.shape) == 2:
                return img
            else:
                raise TypeError('The label image shape dimension is not equal to 2 or 3.')

        try:
            extentions = label_path.split('.')[-1].lower()
            if extentions in ['png', 'tif']:
                label = cv2.imread(label_path, -1)
                label = label_standardization(label)
                return label
            else:
                raise TypeError('The label image type of {} is not supported in training yet.'.format(extentions))
        except:
            raise ValueError('Load label image {} failed.'.format(label_path))


    def _scale_img(self, img, label, scale_img):
        try:
            assert img.shape[0:2] == label.shape[0:2]
        except:
            raise ValueError('The size of source image is not equal to label image.')
        height, width = img.shape[0:2]
        size = (int(height*scale_img), int(width*scale_img))
        img = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
        return img, label


    def _get_single_item(self, index):
        image_path, label_path = self.dataset[index]

        image_path = osp.join(self.source_img_root, image_path)
        label_path = osp.join(self.label_img_root, label_path)

        image = self._img_reader(image_path)
        label = self._label_reader(label_path)

        image, label = self._scale_img(image, label, self.scale_img)
        image, label = self.augment(image, label)

        weight = 0
        # enlarge edges and added weight map
        if self.weighted_type == 'edge_weighted':
            weight = make_weight_map_instance(label, w0=10, sigma=5)
        # balance class
        if self.weighted_type == 'sample_balance':
            weight = make_balance_weight_map(label)

        # binary label
        label[label > 0] = 255
        label = label.astype(np.uint8)
       
        # Transforms
        transform = {
            'image': Compose([ToTensor()]),
            'label': Compose([ToTensor()])
        }
    
        if transform is not None:
            image = transform['image'](image)
            label = transform['label'](label)

        return {"image": image,
                "label": label,
                "weight": weight}


def build_dataloader(name, source_img_root, label_img_root, log_root, validation_ratio, scale_img,
                        weighted_type, aug_list, batch_size, workers, **kwargs):

    dataset = create(name=name, source_img_root=source_img_root, label_img_root=label_img_root, 
                        log_root=log_root, validation_ratio=validation_ratio)

    train_set = dataset.train
    val_set = dataset.val

    train_loader = DataLoader(
        Preprocessor(dataset=train_set, mode='train', 
                        source_img_root=source_img_root,
                        label_img_root=label_img_root,
                        scale_img=scale_img,
                        weighted_type=weighted_type,
                        aug_list=aug_list),
        batch_size=batch_size, 
        num_workers=workers, 
        shuffle=True,
        pin_memory=True, 
        drop_last=True)

    val_loader = DataLoader(
        Preprocessor(dataset=val_set, mode='val', 
                        source_img_root=source_img_root,
                        label_img_root=label_img_root,
                        scale_img=scale_img,
                        weighted_type=weighted_type,
                        aug_list=None),
        batch_size=batch_size, 
        num_workers=workers,
        shuffle=False, 
        pin_memory=True)

    return train_loader, val_loader


