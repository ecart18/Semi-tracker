# -*- coding: UTF-8 -*-

from __future__ import absolute_import

import os.path as osp
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .utils import mkdir


class Visualization():
    def __init__(self, write_folder, add_color=True, add_box=True, 
                    add_edge=True, add_txt=True, 
                    proj_name='', fps=1):
        self._write_folder  =  write_folder
        self._proj_name     =  proj_name
        self.add_color      =  add_color
        self.add_box        =  add_box
        self.add_edge       =  add_edge
        self.add_txt        =  add_txt
        self.fps            =  fps

        self._visual_transformer = []
        if self.add_color:
            self._visual_transformer.append(self._add_color)
        if self.add_box:
            self._visual_transformer.append(self._add_bbox)
        if self.add_edge:
            self._visual_transformer.append(self._add_edge)
        if self.add_txt:
            self._visual_transformer.append(self._add_txt)
        if self._add_trajectory:
            self._visual_transformer.append(self._add_trajectory)
        
        self.video_hander = None


    def _add_frame_index(self, visual_img, frame_index, **kwargs):
        height, width = visual_img.shape[0:2]
        txt = 'Frame: {}'.format(frame_index)
        txt_coor_x, txt_coor_y= min(width, 30), min(height, 30)
        visual_img = cv2.putText(visual_img, txt, (txt_coor_x, txt_coor_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        return visual_img

    def _add_txt(self, visual_img, bbox, label_id, **kwargs):
        txt = '({})'.format(int(label_id))
        visual_img = cv2.putText(visual_img, txt, 
                                (int(0.5*(bbox[0] + bbox[2])), int(0.5*(bbox[1] + bbox[3]))), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        return visual_img
       

    def _add_bbox(self, visual_img, bbox, color, **kwargs):
        visual_img = cv2.rectangle(visual_img, (int(bbox[0]), int((bbox[1]))), (int(bbox[2]), int(bbox[3])), tuple(color), 2)
        return visual_img

    def _add_edge(self, visual_img, edge, **kwargs):
        visual_img[edge[:,1], edge[:,0], :] = np.array([255,255,0])
        return visual_img

    def _add_color(self, visual_img, coords, color, **kwargs):
        visual_img = visual_img.astype(np.float32)
        visual_img[coords[0], coords[1], :] = 0.5 * visual_img[coords[0], coords[1], :] +  0.5 * np.array(color)
        visual_img = visual_img.astype(np.uint8)
        return visual_img
    
    def _add_trajectory(self, visual_img, **kwargs):
        return visual_img

    def _add_attr_per_ins(self, ins, visual_img):
        coords = ins.coords
        color = ins.color
        bbox = ins.bbox
        label_id = ins.label_id
        edge = ins.edge
        for func in self._visual_transformer:
            params = {'coords': coords,
                      'color': color,
                      'bbox': bbox,
                      'label_id': label_id,
                      'edge': edge
            }
            visual_img = func(visual_img=visual_img, **params)
        return visual_img

    def _add_attr_per_frame(self, frame, frame_index):
        raw_img = frame.raw_img
        visual_img = np.copy(raw_img)
        visual_img = self._add_frame_index(visual_img, frame_index)
        for _, ins in frame.instances.items(): 
            visual_img = self._add_attr_per_ins(ins, visual_img)
        return visual_img

    def _to_mp4(self, visual_img, visual_folder):
        if not self.video_hander:
            video_file_name = osp.join(visual_folder, 'visualization_video.mp4')
            height, width = visual_img.shape[0:2]
            self.video_hander = cv2.VideoWriter(video_file_name, 
                                    cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), self.fps, (width, height))
            self.video_hander.write(visual_img)
        else:
            self.video_hander.write(visual_img)


    def __call__(self, frames):
        visual_folder = osp.join(self._write_folder, self._proj_name, 'visual')
        mkdir(visual_folder)
        for frame_index, frame in frames.items():
            visual_img = self._add_attr_per_frame(frame, frame_index)
            visual_img = cv2.cvtColor(visual_img, cv2.COLOR_RGB2BGR)
            visual_save_name = osp.join(visual_folder, osp.basename(frame.file_name))
            cv2.imwrite(visual_save_name, visual_img)
            self._to_mp4(visual_img, visual_folder)
            



