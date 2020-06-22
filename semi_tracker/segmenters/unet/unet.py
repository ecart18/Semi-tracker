# -*- coding: UTF-8 -*-

from __future__ import absolute_import

import cv2
import numpy as np
import torch
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import ToTensor
from .backbone import get_backbone
from .utils import mkdir
from .utils import load_params
from ..utils import bgr_to_gray
from ..utils import gray_to_bgr


class Unet:
    def __init__(self, model_path, threshold=0.5, device='cpu'):
        if torch.cuda.is_available() and device=='gpu':
            self._device = 'cuda'
        else:
            self._device = 'cpu'
        self._model_path = model_path
        self._threshold = threshold
        self._model = self._load_model(device=self._device)
        self.train_transformer = Compose([ToTensor()])

    def _load_model(self, device):
        # Create the model
        model = get_backbone(name='unet', n_channels=1, n_classes=1).to(device).eval()
        model = load_params(model, self._model_path)
        return model

    def _pre_process(self, img):
        if len(img.shape) == 3:
            img = bgr_to_gray(img)
        return self.train_transformer(img).unsqueeze(0)

    def __call__(self, img):
        with torch.no_grad():
            img = self._pre_process(img)
            img = img.to(self._device)
            binary_mask = torch.sigmoid(self._model(img)).squeeze(dim=0).permute(dims=[1,2,0]).cpu().numpy()
            binary_mask = cv2.medianBlur((255 * binary_mask).astype(np.uint8), 3)
            binary_mask = binary_mask.astype(np.float) / 255
            binary_mask[binary_mask < self._threshold] = 0
            binary_mask[binary_mask >= self._threshold] = 1
            binary_mask = np.expand_dims(binary_mask, 2)
            _, label_img, _, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=4, ltype=cv2.CV_32S)        
        return np.expand_dims(label_img, axis=2)
