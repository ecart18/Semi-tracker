# -*- coding: UTF-8 -*-

from __future__ import absolute_import

from collections import OrderedDict
import os.path as osp
import cv2
from jinja2 import Environment, FileSystemLoader 
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .utils import mkdir


class Writer:
    def __init__(self, write_folder, proj_name, cell_type, fps=None, label_txt=True):
        self._write_folder = write_folder
        self._proj_name = proj_name
        self._cell_type = cell_type
        if fps is None:
            self._fps = 'Unknown'
        else:
            self._fps = fps
        self._label_txt = label_txt

    def __call__(self, frames):
        self._visual(frames)
        self._generate_html(frames)

    def _plot(self, frame_index, area, intensity, save_name, cell_id):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        line_intensity = ax1.plot(frame_index, intensity, 'b--o', lw=3, markersize=10, label='Intensity')
        ax1.set_ylabel('Cell Mean Intensity (A.U)', fontdict={'family' : 'Times New Roman', 'size'   : 10})
        ax1.set_title("Cell ID {:0>4d}".format(cell_id))
        ax1.set_xlabel('Frame Index', fontdict={'family' : 'Times New Roman', 'size'   : 10})

        ax2 = ax1.twinx()
        line_area = ax2.plot(frame_index, area, 'r--o', lw=3, markersize=10, label='Area')
        ax2.set_xlim([0, max(frame_index)])
        step = max(max(frame_index) // 10, 1)
        ax2.set_xticklabels(list(range(0, max(frame_index)+step, step)), fontdict={'family' : 'Times New Roman', 'size'   : 16})
        ax2.set_ylabel('Cell Mean Area', fontdict={'family' : 'Times New Roman', 'size'   : 10})
        lines = line_intensity + line_area
        labs = [l.get_label() for l in lines]
        ax1.legend(lines, labs, loc=0, fontsize=10)
        fig.tight_layout()
        mkdir(osp.dirname(save_name))
        plt.savefig(save_name)
        plt.close()
        
    def _generate_cell_info(self, frames):
        cell_info = OrderedDict()
        frame_num = len(frames)
        for frame_index, frame in frames.items():
            instances = frame.instances
            for cell_label, ins in instances.items():
                cell_label = ins.label_id
                if cell_label in cell_info.keys():
                    cell_info[cell_label]['end_frame'] = frame_index
                    cell_info[cell_label]['area'][frame_index] = ins.area
                    cell_info[cell_label]['intensity'][frame_index] = ins.intensity
                else:
                    area = np.zeros(frame_num)
                    area[frame_index] = ins.area
                    intensity = np.zeros(frame_num)
                    intensity[frame_index] = ins.intensity
                    cell_info[cell_label] = {'cell_id':cell_label, 
                                            'start_frame':frame_index, 
                                            'end_frame':frame_index,
                                            'mean_area':0, 
                                            'mean_intensity':0, 
                                            'area': area,
                                            'intensity': intensity,
                                            'image_path':osp.join('./statis', "cell_{:0>4d}.png".format(cell_label))}
        frame_index = np.array(range(frame_num))
        for cell_label in cell_info.keys():
            mean_intensity = cell_info[cell_label]['area'] * cell_info[cell_label]['intensity']
            mean_intensity = np.sum(mean_intensity) / np.sum(cell_info[cell_label]['area'])
            mean_area = np.mean(cell_info[cell_label]['area'][cell_info[cell_label]['area']>0])
            cell_info[cell_label]['mean_area'] = mean_area
            cell_info[cell_label]['mean_intensity'] = mean_intensity
            self._plot(frame_index, cell_info[cell_label]['area'], cell_info[cell_label]['intensity'], 
                            osp.join(self._write_folder, 'statis', "cell_{:0>4d}.png".format(cell_label)),cell_label)
        return cell_info


    def _generate_html(self, frames):
        env = Environment(loader=FileSystemLoader('./'))
        template = env.get_template('./semi_tracker/writer/templates/results.html')
        frame_num = len(frames)
        img_area = '600 * 600' 
        infos = [info for info in self._generate_cell_info(frames).values()]         
        with open(osp.join(self._write_folder, "result.html"),'w+') as html_file:   
            html_content = template.render(proj_name=self._proj_name, 
                                            frame_num=frame_num, fps=self._fps, 
                                            img_area=img_area, cell_type=self._cell_type,
                                            infos=infos)
            html_file.write(html_content)
      
    def _visual(self, frames):
        for frame_index, frame in frames.items():
            save_name = osp.join(self._write_folder, 'visual', osp.basename(frame.file_name))
            raw_img = frame.raw_img
            height, width = raw_img.shape[0:2]
            visual_img = np.copy(raw_img).astype(np.float32)
            for _, ins in frame.instances.items():
                coords = ins.coords
                color = np.ones_like(visual_img) * np.array(ins.color)
                visual_img[coords[0], coords[1], :] = 0.5 * visual_img[coords[0], coords[1], :] + \
                    0.5 * color[coords[0], coords[1], :]
                if self._label_txt:
                    bbox = ins.bbox
                    txt = '({})'.format(int(ins.label_id))
                    visual_img = cv2.putText(visual_img, txt, (int(0.5*(bbox[0] + bbox[2])), int(0.5*(bbox[1] + bbox[3]))), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                    visual_img = cv2.rectangle(visual_img, (int(bbox[0]), int((bbox[1]))), (int(bbox[2]), int(bbox[3])), tuple(ins.color), 2)
                    txt = 'Frame: {}'.format(frame_index)
                    txt_coor_x = max(int(width / 10), 30)
                    txt_coor_y = max(int(height / 10), 30)
                    visual_img = cv2.putText(visual_img, txt, (txt_coor_x, txt_coor_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            # TODO
            mkdir(osp.dirname(save_name))
            cv2.imwrite(save_name, visual_img)