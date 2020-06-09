# -*- coding: UTF-8 -*-
 
from __future__ import absolute_import
import collections
import random
import numpy as np
import cv2
from semi_tracker.utils.logger import logger


class Instance(object):
    """
    The Instance is the basic element of a cell. it contains the following information:
    1) frame index (frame_id, int)
    2) instance label id (label_id, int)
    3) bbox (a box numpy array([l, t, r, b]))
    4) coords, mask coordinates, tuple (x, y) with numpy array 
    5) color, color
    6) name, name in front end 
    """
    def __init__(self, frame_id, label_id, name=None, raw_img=None):
        
        self._frame_id      = frame_id
        self._label_id      = label_id
        self._raw_img       = raw_img

        
        self._coords        = None
        self._bbox          = None
        self._color         = None
        self._area          = None
        self._edge          = None
        self._centroid      = None
        self._intensity     = None

        if name is None:
            self._name        = "cell_" + str(label_id)
        else:
            self._name        = None

    @property
    def intensity(self):
        return self._intensity

    @property
    def raw_img(self):
        return self._raw_img

    @raw_img.setter
    def raw_img(self, raw_img):
        self._raw_img = raw_img

    @property
    def label_id(self):
        return self._label_id
    
    @label_id.setter
    def label_id(self, label_id):
        self._label_id = label_id

    @property
    def frame_id(self):
        return self._frame_id
    
    @frame_id.setter
    def frame_id(self, frame_id):
        self._frame_id = frame_id

    @property
    def centroid(self):
        return self._centroid

    @centroid.setter
    def centroid(self, centroid):
        self._centroid = centroid

    @property
    def area(self):
        return self._area

    @area.setter
    def area(self, area):
        self._area = area

    @property
    def bbox(self):
        return self._bbox

    @bbox.setter
    def bbox(self, bbox):
        self._bbox = bbox

    @property
    def coords(self):
        return self._coords

    @property
    def edge(self):
        return self._edge  # array of [y, x]

    @staticmethod
    def _find_edge_coordinates(x_max, y_max, coords):
        binary_mask = np.zeros((y_max + 1, x_max + 1), dtype=np.uint8)
        binary_mask[coords[0], coords[1]] = 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        edge = np.vstack((x.reshape(-1,2) for x in contours))  # array of [y, x]
        return edge

    @coords.setter
    def coords(self, coords):
        self._coords = coords
        # bbox setting
        x1, x2 = np.min(self._coords[1]), np.max(self._coords[1])
        y1, y2 = np.min(self._coords[0]), np.max(self._coords[0])
        self._bbox = np.array([x1, y1, x2, y2])
        self._centroid = [int((x1+x2)/2), int((y1+y2)/2)]
        self._area = np.shape(self._coords)[1]
        self._intensity = np.sum(self._raw_img[coords[0], coords[1], :]) / self._area
        self._edge = self._find_edge_coordinates(x_max=x2, y_max=y2, coords=coords)
        

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, color):
        self._color = color

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    # def setInstanceId(self, ins_id):
    #     self.instance_id = ins_id

    # def setCoords(self, coords):
        # self.coords = coords

    # def setColor(self, color):
    #     self.color = color

    # def setName(self, name):
    #     self.name = name


class Frame(object):
    """
    The Frame is the basic element of a video. it contains the following information:
    1) frame index (frame_id, int)
    2) file name (file_name, str)
    3) raw_img (numpy array height * width * 3)
    """
    def __init__(self, file_name, frame_id, raw_img):
        self._frame_id      = frame_id
        self._file_name     = file_name
        self._raw_img       = raw_img
        self._img_size      = np.shape(raw_img)[0:-1]
        self._norm_img      = raw_img

        self._binary_mask   = None   # binary mask (height, width, 1)
        self._label_img     = None   # label (height, width, 1)
        self._raw_color_img = raw_img   # color and raw img
        
        self._label_n       = 0
        self._label_max     = 0
        self._labels        = []
        self._color_map     = []
        self._color_map_dict = collections.OrderedDict()
        self._color_map.append([54, 54, 54])
        # self._color_map[0] = [54, 54, 54]  # background color
        self._instances    = collections.OrderedDict()                 # hash map for id:instance

    @property
    def labels(self):
        return self._labels

    @property
    def label_max(self):
        return self._label_max

    @property
    def norm_img(self):
        return self._norm_img

    @norm_img.setter
    def norm_img(self, norm_img):
        self._norm_img = norm_img

    @property
    def label_n(self):
        return self._label_n

    @property
    def frame_id(self):
        return self._frame_id
    
    @frame_id.setter
    def frame_id(self, frame_id):
        self._frame_id = frame_id

    @property
    def file_name(self):
        return self._file_name

    @file_name.setter
    def file_name(self, file_name):
        self._file_name = file_name
    
    @property
    def raw_img(self):
        return self._raw_img

    @raw_img.setter
    def raw_img(self, raw_img):
        self._raw_img = raw_img

    @property
    def binary_mask(self):
        return self._binary_mask

    @binary_mask.setter
    def binary_mask(self, binary_mask):
        try:
            assert np.shape(binary_mask)[0:-1] == self._img_size
        except AssertionError:
            logger.error('The size of binary_mask is not compatible with raw image.')
        self._binary_mask = binary_mask

    @property
    def raw_color_img(self):
        return self._raw_color_img

    @property
    def color_map_dict(self):
        return self._color_map_dict

    @property
    def label_img(self):
        return self._label_img
    
    @label_img.setter
    def label_img(self, label_img):
        try:
            assert np.shape(label_img)[0:-1] == self._img_size
        except AssertionError:
            logger.error('The size of label image is not compatible with raw image.')
        self._label_img = label_img.astype(np.int16)
        self._binary_mask = self._label_img > 0
        self._binary_mask = 255 * self._binary_mask
        self._binary_mask = self._binary_mask.astype(np.uint8)
        self._labels = np.delete(np.unique(self._label_img), 0)
        self._label_max = label_img.max()
        

    @property
    def color_map(self):
        return self._color_map

    @color_map.setter
    def color_map(self, color_map):
        self._color_map = color_map

    @property
    def instances(self):
        return self._instances

    def add_instance(self, label, instance):
        self._instances[label] = instance
        self._color_map.append(instance.color)
        # self._color_map[label] = instance.color
        self._label_max += 1
        self._label_n += 1

    def auto_labeling1(self, color_groups, color_map_dict):
        self._color_map = []
        self._color_map.append([54, 54, 54])
        self._color_map_dict = collections.OrderedDict()
        self._instances = collections.OrderedDict()
        self._raw_color_img = np.copy(self._raw_img).astype(np.float32)

        # print(color_map_dict)
        if color_map_dict is None:
            self.auto_labling(color_groups)
        else:
            labels = np.delete(np.unique(self._label_img), 0)
            # print(labels)
            self._label_max = labels.max()
            self._label_n = len(labels)
            for idx, label in enumerate(labels):
                if int(label) in color_map_dict.keys():
                    # print(int(label))
                    color = color_map_dict[int(label)]
                    self._color_map_dict[int(label)] = color
                    # print(color)
                else:
                    r = random.randint(0, 11)
                    c_color = color_groups[idx % 26]
                    color = c_color[r]
                    self._color_map_dict[int(label)] = color
                self._color_map.append(color)
                # self._color_map[label] = c_color[r]
                instance = Instance(self._frame_id, label_id=label, raw_img=self._raw_img)
                instance.color = color
                instance.coords = np.where(self._label_img == label)

                color = np.ones_like(self._raw_color_img) * np.array(instance.color)
                coords = instance.coords
                self._raw_color_img[coords[0], coords[1], :] = (0.5 * self._raw_color_img[coords[0], coords[1], :] +
                                                                0.5 * color[coords[0], coords[1], :])

                self._instances[label] = instance

    def auto_labling(self, color_groups):
        labels = np.delete(np.unique(self._label_img), 0)
        # print(labels)
        self._label_max = labels.max()
        self._label_n = len(labels)
        self._raw_color_img = np.copy(self._raw_img).astype(np.float32)

        for idx, label in enumerate(labels):
            r = random.randint(0, 11)
            c_color = color_groups[idx % 26]
            self._color_map.append(c_color[r])
            self._color_map_dict[int(label)] = c_color[r]
            instance = Instance(self._frame_id, label_id=label, raw_img=self._raw_img)
            instance.color = c_color[r]
            instance.coords = np.where(self._label_img == label)

            color = np.ones_like(self._raw_color_img) * np.array(instance.color)
            coords = instance.coords
            self._raw_color_img[coords[0], coords[1], :] = (0.5 * self._raw_color_img[coords[0], coords[1], :] +
                                                            0.5 * color[coords[0], coords[1], :])
            self._instances[label] = instance

    def update_labling(self, update_ins_id=None, update_ins_label=None, update_ins_name=None,
                       update_ins_color=None, delete_ins_id=None, delete_ins_label=None, annotation_flag=0):
        """
        add, delete and modify instances
        """
        # print(update_ins_label)
        if update_ins_label is not None:
            for ins_id, ins_label, ins_name, ins_color in \
                    zip(update_ins_id, update_ins_label, update_ins_name, update_ins_color):
                instance = Instance(frame_id=self.frame_id, label_id=ins_label, name=ins_name, raw_img=self.raw_img)
                self._color_map[ins_id+1] = ins_color
                instance.color = ins_color
                instance.coords = np.where(self._label_img == ins_label)
                instance.name = ins_name

                color = np.ones_like(self._raw_color_img) * np.array(instance.color)
                coords = instance.coords
                # print(np.shape(self.raw_img))
                # print(np.shape(color))
                self._raw_color_img[coords[0], coords[1], :] = (0.5 * self.raw_img[coords[0], coords[1], :] +
                                                                0.5 * color[coords[0], coords[1], :])
                self._instances[ins_label] = instance
        if delete_ins_label is not None:
            for ins_id, ins_label in zip(delete_ins_id, delete_ins_label):
                self._color_map.remove(self._color_map[ins_id+1])
                ins = self.instances[ins_label]
                if annotation_flag == 0:
                    self._raw_color_img[ins.coords[0], ins.coords[1], :] = self.raw_img[ins.coords[0], ins.coords[1], :]
                if ins.coords is not None:
                    if annotation_flag == 2:
                        self._label_img[ins.coords[0], ins.coords[1]] = 0
                    else:
                        self._label_img[ins.coords[0], ins.coords[1], ins.coords[2]] = 0
                self._label_n -= 1
                self._instances.pop(ins_label)
    
    def add_labeling(self, update_ins_id=None, update_ins_label=None, update_ins_name=None,
                     update_ins_color=None):
        if not len(update_ins_label) == 0:
            for ins_id, ins_label, ins_name, ins_color in \
                    zip(update_ins_id, update_ins_label, update_ins_name, update_ins_color):
                instance = Instance(frame_id=self.frame_id, label_id=ins_label, name=ins_name, raw_img=self.raw_img)
                self._color_map[ins_id+1] = ins_color
                instance.color = ins_color
                instance.coords = np.where(self._label_img == ins_label)
                instance.name = ins_name
                self._instances[ins_label] = instance

    # def updateLabelImg(self, label_img):
    #     self.label_img = label_img

    # def setLabelImg(self, label_img, my_colors):
    #     self.label_img = label_img
    #     self.label_n = self.label_img.max()
    #     self.color_map = []
    #     self.color_map.append([54, 54, 54])
    #     for i in range(self.label_n):
    #         r = random.randint(0, 11)
    #         t = i % 26
    #         # print(t)
    #         c_color = my_colors[t]
    #         self.color_map.append(c_color[r])

    # def setColorMap(self, color_map):
    #     self.color_map = color_map

    # def getInstances(self):
    #     self.label_n = self.label_img.max()
    #     for i in range(self.label_n):
    #         ins = Instance(self.frame_id, label_id=i+1)
    #         name = "cell_" + str(i+1)
    #         ins.name = name
    #         ins.color = self.color_map[i+1]
    #         ins.coords = np.where(self.label_img == i+1)

    #         minx = self.img_size[0]
    #         maxx = 0
    #         miny = self.img_size[1]
    #         maxy = 0
    #         for c in ins.coords:
    #             if c[0] < minx:
    #                 minx = c[0]
    #             if c[1] < miny:
    #                 miny = c[1]
    #             if c[0] > maxx:
    #                 maxx = c[0]
    #             if c[1] > maxy:
    #                 maxy = c[1]

    #         ins.bbox = np.array([minx, miny, maxx, maxy])

    #         self.instances[i + 1] = ins

    # def addInstances(self, ins_id, ins):
    #     self.instances[ins_id] = ins

    # def deleteInstance(self, ins_id):
    #     self.color_map.pop(ins_id)
    #     ins = self.instances[ins_id]
    #     for i in ins.coords:
    #         self.label_img[i[0], i[1], 0] = 0
    #     self.label_n -= 1
    #     self.instances.pop(ins_id)
    #     temp = {}
    #     for key in self.instances.keys():
    #         if key > ins_id:
    #             print(key)
    #             temp_ins = self.instances[key]
    #             for j in temp_ins.coords:
    #                 self.label_img[j[0], j[1], 0] -= 1
    #             temp_ins.label_id = key-1
    #             nm = "cell_" + str(key-1)
    #             temp[key-1] = temp_ins
    #         else:
    #             temp[key] = self.instances[key]

    #     self.instances = temp


