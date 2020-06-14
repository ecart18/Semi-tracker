# -*- coding: UTF-8 -*-

import collections
import os
import sys
import numpy as np
import cv2
import imageio
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import QSize, Qt
from PyQt5 import QtCore, QtWidgets
from PyQt5 import QtGui
import pyqtgraph as pg
from .config import SYSTEM
from .components import CorrectionTools
from .components import ChangedInstances
from .components import CellAttributeWindow
from .components import FileTree
from .components import InstanceSettings
from .components import LeftNavigation
from .components import LeftTools
from .components import Menu
from .components import ProjectWindow
from .components import StatusBar
from .components import VisualizeWindow
from .components import QuestionMessageBox, InformationMessageBox, WarningMessageBox
from .components import AnnotationTools
from .utils import color_groups, annotation_colors
from .utils import load_images
from .utils import mkdir
from ..normalizers import get_normalizer
from ..segmenters import get_segmenter
from ..trackers import get_tracker
from ..writer import get_writer
from .utils import get_icon, slide_stylesheet, general_qss
from ..utils import logger
from semi_tracker import PACKAGEPATH


class MainWindow(QMainWindow):
    # MainWindow
    # contains:
    #

    def __init__(self, parent=None):
        super().__init__()

        pg.setConfigOption('imageAxisOrder', 'row-major')

        mkdir(os.path.join(PACKAGEPATH, "../output"))
        mkdir(os.path.join(PACKAGEPATH, "../checkpoint"))
        self.project_path       = os.path.join(PACKAGEPATH, "../output/untitled")
        self.unet_model_path    = os.path.join(PACKAGEPATH, "../checkpoint/model_best.pth.tar")
        self.last_index         = 0             # left navigation update
        self.frames_num         = 0             # num of frames
        self.frames             = collections.OrderedDict()
        self.show_flag          = 1    # 1:raw raw 2.mask raw 3.color+raw raw 4.annotation: raw label_img
        # 0:unload 1.loaded 2. segmented 3.tracked 4.annotation_origin_selected 5.annotation_ready
        # self.status_flag        = 0
        self.segmenter_dict     = {0: 'binary_thresholding', 1: 'unet', 2: 'water_shed', 3: 'grab_cut', 4: 'User-defined'}
        self.segmenter_name     = 'binary_thresholding'
        self.tracker_dict       = {0: 'none', 1: 'bipartite_tracker'}
        self.tracker_name       = 'bipartite_tracker'
        self.normalizer_dict    = {0: 'equalize_hist', 1: 'min_max', 2: 'retinex_MSRCP', 3: 'retinex_MSRCR'}
        self.normalizer_name    = 'equalize_hist'
        self.previous_frame     = 0

        self.grabcut_pos_begin  = None
        self.grabcut_pos_finish = None
        self.grabcut_roi        = None
        self.grabcut_rect       = None

        # annotation
        self.annotation_flag    = -1        # -1 none 0 modify 1 modify annotation 2 annotation
        self.frames_roi         = {}
        self.widget_list        = []       # CurrentRow->label_id
        self.changed_instances  = ChangedInstances()
        self.bboxroi            = None
        self.color_roi          = []
        self.drag_flag          = 0    # 0 ignore 1 dragging
        self.background_color   = [54, 54, 54]
        self.my_colors          = color_groups

        self.result_dir_path    = ""
        self.origin_dir_path    = ""

        self.sld_stylesheet = slide_stylesheet
        self.general_qss = general_qss
        self.setStyleSheet(self.general_qss)

        self.setup_ui()

    def setup_ui(self):

        # Initialize five parts of the main window
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        self.main_widget = main_widget

        self.init_menu()
        self.init_left_part()
        self.init_visualize_window()
        self.init_correction_tools()
        self.init_status_bar()


        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.left_navigation)
        main_layout.addWidget(self.left_tools)
        main_layout.addWidget(self.visualize_window)
        main_layout.addWidget(self.correction_tools)
        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 3)
        main_layout.setStretch(2, 12)
        main_layout.setStretch(3, 4)

        self.setCentralWidget(main_widget)


        self.setGeometry(300, 300, 1500, 800)
        self.setWindowTitle("SemiTracker")
        self.setWindowIcon(QIcon(get_icon("cell.png")))
        self.setStatusBar(self.status_bar)
        self.center()
        self.show()

    def init_menu(self):
        # Initialize menu bar
        self.menu = Menu()
        self.menu_bar = self.menu.menu_bar

        self.menu.new_project_act.triggered.connect(self.new_project_fnc)
        self.menu.load_act.triggered.connect(self.load_fnc)
        self.menu.exit_act.triggered.connect(self.exit_fnc)

        self.setMenuBar(self.menu_bar)

    def init_left_part(self):
        self.navigation = LeftNavigation()
        self.left_navigation = self.navigation.left_navigation
        self.tools = LeftTools()
        self.left_tools = self.tools.left_tools
        self.tools.main_algorithm.currentChanged.connect(self.main_algorithm_changed_fnc)

        self.left_navigation.clicked.connect(self.left_navigation_fnc)
        self.last_item = self.left_navigation.item(self.last_index)
        self.last_item.setBackground(QColor("#323232"))

        # file_tree
        self.tools.file_tree_widget.create_button.clicked.connect(self.new_project_fnc)
        # self.tools.file_tree_widget.open_button.clicked.connect(self.open_project_fnc)

        self.tools.segment.segment_tools.currentChanged.connect(self.segmenter_changed_fnc)
        self.tools.normlize.normalize_tools.currentChanged.connect(self.normalizer_changed_fnc)

        # seg alg1
        self.tools.segment.thresh_segment_button.clicked.connect(
            lambda: self.segment(self.segmenter_name, threshold=self.tools.segment.thresh_sld1.value()))

        # seg alg2
        self.tools.segment.model_browse_button.clicked.connect(self.model_select_fnc)
        self.tools.segment.unet_segment_button.clicked.connect(
            lambda: self.segment(self.segmenter_name, model_path=self.unet_model_path,
                                 threshold=self.tools.segment.thresh_sld2.value()/10))

        # seg alg3
        self.tools.segment.watershed_segment_button.clicked.connect(
            lambda: self.segment(self.segmenter_name, noise_amplitude=self.tools.segment.noise_sld.value(),
                                 dist_thresh=self.tools.segment.dist_thresh_sld.value()/10))

        # seg alg4
        self.tools.segment.select_roi_button.clicked.connect(self.select_roi_button_fnc)
        self.tools.segment.grabcut_segment_button.clicked.connect(
            lambda: self.segment(self.segmenter_name, iteration=self.tools.segment.iteration_sld.value()))

        # normalization
        self.tools.normlize.equalize_hist_button.clicked.connect(lambda: self.normalize(self.normalizer_name))
        self.tools.normlize.min_max_button.clicked.connect(lambda: self.normalize(self.normalizer_name))
        self.tools.normlize.retinex_MSRCP_button.clicked.connect(lambda: self.normalize(self.normalizer_name))
        self.tools.normlize.retinex_MSRCR_button.clicked.connect(lambda: self.normalize(self.normalizer_name))


        # track
        self.tools.track.run_button.clicked.connect(lambda: self.track(self.tracker_name))

        # output
        self.tools.output.output_button.clicked.connect(lambda: self.write(write_folder=self.project_path))

        # annotation
        self.tools.annotation.origin_path_browse_button.clicked.connect(self.dir_loader)
        self.tools.annotation.result_path_browse_button.clicked.connect(self.result_path_fnc)
        self.tools.annotation.save_annotation_button.clicked.connect(self.save_annotation_fnc)
        self.tools.annotation.finish_annotation_button.clicked.connect(self.finish_annotation_fnc)

    def init_visualize_window(self):
        self.visualize = VisualizeWindow()
        self.visualize_window = self.visualize.visualize_window
        # self.main_widget.setMouseTracking(True)
        self.visualize_window.setMouseTracking(True)
        # self.setMouseTracking(True)
        self.visualize.setMouseTracking(True)
        # self.visualize.main_frame.setMouseTracking(True)
        self.visualize.main_frame.view.mouseMoveEvent = self.mouseMoveEvent
        self.visualize.main_left.clicked.connect(self.main_left_fnc)
        self.visualize.main_right.clicked.connect(self.main_right_fnc)
        self.visualize.main_sld.valueChanged.connect(self.main_sld_fnc)

    def init_correction_tools(self):
        self.correction = CorrectionTools()
        self.correction_tools = self.correction.right_widget
        # self.correction_tools.setEnabled(False)

        self.correction.add_instance_button.clicked.connect(self.add_instance_fnc)
        self.correction.instances_widget.clicked.connect(self.instances_widget_fnc)
        self.correction.instances_widget.doubleClicked.connect(self.instances_widget_fnc_double)
        self.correction.size_editor.textChanged.connect(self.size_editor_fnc)
        self.correction.size_left_button.clicked.connect(self.size_left_fnc)
        self.correction.size_right_button.clicked.connect(self.size_right_fnc)
        # self.correction.switch_show_button.clicked.connect(self.switch_show_fnc)
        self.correction.brush_button.clicked.connect(self.brush_fnc)
        self.correction.eraser_button.clicked.connect(self.eraser_fnc)
        self.correction.drag_button.clicked.connect(self.drag_fnc)
        self.correction.confirm_button.clicked.connect(self.confirm_fnc)
        self.correction.delete_button.clicked.connect(self.instance_delete_fnc)

    def init_status_bar(self):
        # Initialize status bar
        self.status = StatusBar()
        self.status_bar = self.status.status_bar

    def mouseMoveEvent(self, event):
        self.setMouseTracking(True)
        # event.accept()
        s = event.windowPos()
        self.status.work_info_label.setText(str(s.x()))
        print(s)

    def select_roi_button_fnc(self):
        self.visualize.main_frame.getImageItem().mouseDragEvent = self.grabcut_drag

    def grabcut_drag(self, event):
        event.accept()
        pos = event.pos()
        if event.isStart():
            self.grabcut_pos_begin = pos
            # self.grabcut_rect[0] = int(pos[0])
        elif event.isFinish():
            self.grabcut_pos_finish = pos
            event.ignore()
        else:
            if self.grabcut_roi is not None:
                self.visualize.main_frame.removeItem(self.grabcut_roi)
            self.grabcut_roi = pg.RectROI(self.grabcut_pos_begin, pos-self.grabcut_pos_begin)
            self.visualize.main_frame.addItem(self.grabcut_roi)

    def main_algorithm_changed_fnc(self):
        algorithm_key = self.tools.main_algorithm.currentIndex()
        for i in range(4):
            if i == algorithm_key:
                self.tools.main_algorithm.setItemIcon(i, QIcon(get_icon("Arrow_down.png")))
            else:
                self.tools.main_algorithm.setItemIcon(i, QIcon(get_icon("Arrow_right.png")))

    def normalizer_changed_fnc(self):
        normalizer_key = self.tools.normlize.normalize_tools.currentIndex()
        for i in range(4):
            if i == normalizer_key:
                self.tools.normlize.normalize_tools.setItemIcon(i, QIcon(get_icon("Arrow_down.png")))
            else:
                self.tools.normlize.normalize_tools.setItemIcon(i, QIcon(get_icon("Arrow_right.png")))
        self.normalizer_name = self.normalizer_dict[normalizer_key]

    def segmenter_changed_fnc(self):
        segmenter_key = self.tools.segment.segment_tools.currentIndex()
        for i in range(5):
            if i == segmenter_key:
                self.tools.segment.segment_tools.setItemIcon(i, QIcon(get_icon("Arrow_down.png")))
            else:
                self.tools.segment.segment_tools.setItemIcon(i, QIcon(get_icon("Arrow_right.png")))
        # print(segmenter_key)
        self.segmenter_name = self.segmenter_dict[segmenter_key]
        # print(self.segmenter_name)

    def model_select_fnc(self):
        model_path = QFileDialog.getOpenFileName(None, "Select a model file..",
                                                 filter="*.tar")[0]
        # print(model_path)
        if model_path == "":
            self.tools.segment.model_path_label.setText("Select a model..")
        else:
            self.tools.segment.model_path_label.setText(model_path)
            self.unet_model_path = model_path

    def main_left_fnc(self):
        if not len(self.frames) == 0:
            v = self.visualize.main_sld.value()
            if v > 0:
                self.visualize.main_sld.setValue(v-1)
                # self.main_sld_fnc()
                ml = " Current frame: " + str(v)
                self.status.frame_info_label.setText(ml)

    def main_right_fnc(self):
        if not len(self.frames) == 0:
            v = self.visualize.main_sld.value()
            if v < self.frames_num:
                self.visualize.main_sld.setValue(v+1)
                # self.main_sld_fnc()
                ml = " Current frame: " + str(v + 2)
                self.status.frame_info_label.setText(ml)

    def main_sld_fnc(self):
        if not len(self.frames) == 0:
            main_sld_val = self.visualize.main_sld.value()
            main_sld_val_label = " Current frame: " + str(main_sld_val)
            self.status.total_frame_label.setText(main_sld_val_label)
            if not np.shape(len(self.frames)) == 0:

                if self.show_flag == 1:
                    self.visualize.main_frame.setImage(self.frames[main_sld_val].norm_img,
                                                       autoHistogramRange=False)
                    self.correction.assist_frame.setImage(self.frames[main_sld_val].raw_img,
                                                          autoHistogramRange=False)

                elif self.show_flag == 2:
                    self.visualize.main_frame.setImage(self.frames[main_sld_val].binary_mask,
                                                       autoHistogramRange=False)
                    self.correction.assist_frame.setImage(self.frames[main_sld_val].raw_img,
                                                          autoHistogramRange=False)

                elif self.show_flag == 3:
                    self.visualize.main_frame.removeItem(self.bboxroi)
                    self.visualize.main_frame.setImage(self.frames[main_sld_val].raw_color_img)
                    self.correction.assist_frame.setImage(self.frames[main_sld_val].raw_img,
                                                          autoHistogramRange=False)
                    self.instance_widget_update_fnc()
                elif self.show_flag == 4:
                    self.visualize.main_frame.removeItem(self.bboxroi)

                    if not len(self.frames_roi[self.previous_frame]) == 0:
                        for c_roi in self.frames_roi[self.previous_frame]:
                            self.visualize.main_frame.removeItem(c_roi)
                    if not len(self.frames_roi[main_sld_val]) == 0:
                        for c_roi in self.frames_roi[main_sld_val]:
                            self.visualize.main_frame.addItem(c_roi)

                    self.visualize.main_frame.setImage(self.frames[main_sld_val].norm_img,
                                                       autoHistogramRange=False)

                    if self.frames[main_sld_val].label_img.min == \
                            self.frames[main_sld_val].label_img.max:
                        colormap = pg.ColorMap(np.linspace(0, 1, self.frames[main_sld_val].label_n + 1),
                                               color=self.frames[main_sld_val].color_map)
                        self.correction.assist_frame.setColorMap(colormap)
                        self.correction.assist_frame.setImage(self.frames[main_sld_val].label_img,
                                                              autoHistogramRange=False)
                    else:
                        colormap = pg.ColorMap(np.linspace(0, 1, 2),
                                               color=[[0, 0, 0, 0], [255, 255, 255, 255]])
                        self.correction.assist_frame.setColorMap(colormap)
                        self.correction.assist_frame.setImage(self.frames[main_sld_val].norm_img)
                    self.instance_widget_update_fnc()
            else:
                pass

            self.previous_frame = main_sld_val

    def instance_widget_update_fnc(self):
        self.widget_list = []
        self.correction.instances_widget.clear()
        for key in self.frames[self.visualize.main_sld.value()].instances.keys():
            show_contents = "Name: " + self.frames[self.visualize.main_sld.value()].instances[key].name + "\n" + \
                            " Centroid: " + str(self.frames[self.visualize.main_sld.value()].instances[key].centroid)
            item = QListWidgetItem(show_contents, self.correction.instances_widget)
            pix = QPixmap(30, 30)
            pix.fill(QColor(self.frames[self.visualize.main_sld.value()].instances[key].color[0],
                            self.frames[self.visualize.main_sld.value()].instances[key].color[1],
                            self.frames[self.visualize.main_sld.value()].instances[key].color[2]))
            item.setIcon(QIcon(pix))
            item.setSizeHint(QSize(35, 35))
            self.widget_list.append(self.frames[self.visualize.main_sld.value()].instances[key].label_id)
            item.setTextAlignment(Qt.AlignCenter)
            self.correction.instances_widget.addItem(item)

    def new_project_fnc(self):
        self.new_project_window = ProjectWindow(self.project_path)
        self.new_project_window.create1.clicked.connect(self.set_project_path_fnc)
        self.new_project_window.show()

    def result_path_fnc(self):
        if self.origin_dir_path == "":
            self.origin_first_message_box = InformationMessageBox("Please select the origin images path first!")
            self.origin_first_message_box.show()
        else:
            self.result_dir_path = QFileDialog.getExistingDirectory()
            self.tools.annotation.result_path_show_lineedit.setText(self.result_dir_path)
            for key in self.frames.keys():
                self.frames[key].label_img = np.zeros(np.shape(self.frames[key].raw_img[:, :, 0]))
                self.frames[key].label_img.astype(np.uint16)
                # print(self.frames[key].label_img.dtype)
                self.frames_roi[key] = []
            self.annotation_flag = 2
            self.show_flag = 4


    # TODO
    def set_project_path_fnc(self):
        mkdir(self.new_project_window.path)
        self.project_path = self.new_project_window.path

        self.tools.update_file_tree(self.project_path)

    def load_fnc(self):
        if len(self.frames) == 0:
            self.files_loader()

        else:
            self.reload_question = QuestionMessageBox("Are you sure to reload the data?")
            self.reload_question.show()
            self.reload_question.yes_button.clicked.connect(self.reload_fnc)
            '''
            reply = QMessageBox.question(self, 'Warning',
                                         "<font color=black>Are you sure to reload the data?</font>",
                                         QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel)
            if reply == QMessageBox.No:
                pass
            elif reply == QMessageBox.Yes:
                self.visualize.main_frame.clear()
                colormap = pg.ColorMap([0, 1], color=[[0, 0, 0], [255, 255, 255]])
                self.visualize.main_frame.setColorMap(colormap)

                self.correction.assist_frame.clear()
                self.correction.assist_frame.setColorMap(colormap)

                self.correction.instances_widget.clear()

                self.files_loader()
            '''
    def reload_fnc(self):
        self.annotation_flag = -1
        self.visualize.main_frame.clear()
        # colormap = pg.ColorMap([0, 1], color=[[0, 0, 0], [255, 255, 255]])
        # self.visualize.main_frame.setColorMap(colormap)

        self.correction.assist_frame.clear()
        # self.correction.assist_frame.setColorMap(colormap)

        self.correction.instances_widget.clear()

        self.reload_question.close_fnc()
        self.files_loader()


    def files_loader(self):
        filenames = QFileDialog.getOpenFileNames(None, "Select lsm data files to concatenate...",
                                                 filter="*bmp *.tif *.png *.jpg *.JPEG *avi *mp4 *mpg")[0]
        if not filenames == []:
            self.frames = load_images(filenames)
            self.frames_num = len(self.frames)

            # colormap = pg.ColorMap([0, 1], color=[[0, 0, 0], [255, 255, 255]])
            # self.visualize.main_frame.setColorMap(colormap)

            self.visualize.main_frame.setImage(img=self.frames[0].raw_img)
            self.correction.assist_frame.setImage(self.frames[0].raw_img)

            self.show_flag = 1
            self.visualize.main_sld.setValue(0)
            self.visualize.main_sld.setMaximum(self.frames_num - 1)
            self.status.frame_info_label.setText(" Current frame: " + str(1))
            self.status.total_frame_label.setText(" Total frames: " + str(self.frames_num))
            # self.status_signal[0] = 1

    def dir_loader(self):
        self.origin_dir_path = QFileDialog.getExistingDirectory()
        self.tools.annotation.origin_path_show_lineedit.setText(self.origin_dir_path)

        # colormap = pg.ColorMap([0, 1], color=[[0, 0, 0], [255, 255, 255]])
        # self.visualize.main_frame.setColorMap(colormap)

        filenames = []
        if not self.origin_dir_path == "":
            for name in os.listdir(self.origin_dir_path):
                file_path = os.path.join(self.origin_dir_path, name)
                filenames.append(file_path)
        if not filenames == []:
            self.frames = load_images(filenames)
            self.frames_num = len(self.frames)
            self.visualize.main_frame.setImage(self.frames[0].raw_img)
            self.correction.assist_frame.setImage(self.frames[0].raw_img)

            self.show_flag = 1
            self.visualize.main_sld.setValue(0)
            self.visualize.main_sld.setMaximum(self.frames_num - 1)
            self.status.frame_info_label.setText(" Current frame: " + str(1))
            self.status.total_frame_label.setText(" Total frames: " + str(self.frames_num))

    def exit_fnc(self):
        self.quit_message_box = QuestionMessageBox('You sure to exit?')
        self.quit_message_box.show()

        self.quit_message_box.yes_button.clicked.connect(self.exit_fnc_yes)

    def exit_fnc_yes(self):
        qApp.quit()

    def save_annotation_fnc(self):
        # print(self.annotation_flag)
        if not self.annotation_flag == 2:
            self.annotation_first_message_box = InformationMessageBox("Please do annotation first!")
            self.annotation_first_message_box.show()
        elif self.annotation_flag == 2:
            for key in self.frames.keys():
                name = "annotation_" + str(key) + ".tif"
                annotation_path = os.path.join(self.result_dir_path, name)
                # print(self.frames[key].label_img.dtype)
                imageio.imwrite(annotation_path, self.frames[key].label_img.astype(np.uint16))
                # print(self.frames[key].label_img.max())
                # cv2.imwrite(annotation_path, self.frames[key].label_img)
            self.annotation_save_message_box = InformationMessageBox("The result has been successfully saved.")
            self.annotation_save_message_box.show()

    def finish_annotation_fnc(self):
        if not self.annotation_flag == 2:
            self.annotation_first_message_box = InformationMessageBox("Please do annotation first!")
            self.annotation_first_message_box.show()
        else:
            self.annotation_finish_message_box = \
                QuestionMessageBox("Are you sure to finish the annotation?")
            self.annotation_finish_message_box.show()
            self.annotation_finish_message_box.yes_button.clicked.connect(self.annotation_finish_message_box_yes_fnc)

    def annotation_finish_message_box_yes_fnc(self):
        self.annotation_flag = -1
        for key in self.frames.keys():
            name = "annotation_" + str(key) + ".tif"
            annotation_path = os.path.join(self.result_dir_path, name)
            imageio.imwrite(annotation_path, self.frames[key].label_img.astype(np.uint16))
            # cv2.imwrite(annotation_path, self.frames[key].label_img)

        '''
        # remove all rois
        if self.bboxroi is not None:
            self.visualize.main_frame.removeItem(self.bboxroi)
        for key in self.frames.keys():
            if key in self.frames_roi.keys():
                for roi in self.frames_roi[key]:
                    self.visualize.main_frame.removeItem(roi)
        
        self.frames_roi = {}
        '''

        # show result colormap image

        self.annotation_finish_message_box.close_fnc()

    def brush_fnc(self):
        if len(self.frames) == 0:
            self.brush_message_box = InformationMessageBox("Please load images first!")
            self.brush_message_box.show()
        elif not len(self.frames) == 0 and self.frames[0].label_img is None and self.annotation_flag == 0:
            self.brush_message_box = InformationMessageBox("Please do segmentation first!")
            self.brush_message_box.show()
        elif self.annotation_flag == 1 or self.annotation_flag == 2:
            pix = QPixmap(get_icon("pen1.png"))
            cursor = QCursor(pix)
            self.visualize_window.setCursor(cursor)
            current_frame = self.frames[self.visualize.main_sld.value()]
            current_id = self.correction.instances_widget.currentRow()
            if current_id >= 0:
                colormap = pg.ColorMap([0, 1], color=[[0, 0, 0], [255, 255, 255]])
                self.visualize.main_frame.setColorMap(colormap)
                self.visualize.main_frame.setImage(current_frame.raw_img[:, :, 0])
                self.visualize.main_frame.getImageItem().mouseDragEvent = self.brush_drag1
            else:
                self.brush_message_box = InformationMessageBox("Please add a cell first!")
                self.brush_message_box.show()
        elif self.annotation_flag == 0:
            pix = QPixmap(get_icon("pen1.png"))
            cursor = QCursor(pix)
            self.visualize_window.setCursor(cursor)
            self.show_flag = 4
            current_frame = self.frames[self.visualize.main_sld.value()]
            current_id = self.correction.instances_widget.currentRow()
            if current_id >= 0:
                current_color = current_frame.instances[self.widget_list[current_id]].color
                current_coords = current_frame.instances[self.widget_list[current_id]].coords
                x = current_coords[0]
                y = current_coords[1]
                for i in range(len(x)):
                    my_roi = pg.ROI(pos=(y[i]-0.5, x[i]-0.5),
                                    pen=QPen(QColor(current_color[0], current_color[1], current_color[2])),
                                    movable=False)
                    self.color_roi.append(my_roi)
                    self.visualize.main_frame.addItem(my_roi)

                colormap = pg.ColorMap([0, 1], color=[[0, 0, 0], [255, 255, 255]])
                self.visualize.main_frame.setColorMap(colormap)
                self.visualize.main_frame.imageItem.updateImage(current_frame.raw_img[:, :, 0])
                self.visualize.main_frame.getImageItem().mouseDragEvent = self.brush_drag1
            else:
                self.brush_message_box = InformationMessageBox("Please add a cell first!")
                self.brush_message_box.show()

    def eraser_fnc(self):
        if len(self.frames) == 0:
            self.eraser_message_box = InformationMessageBox("Please load images first!")
            self.eraser_message_box.show()
        elif not len(self.frames) == 0 and self.frames[0].label_img is None and self.annotation_flag == 0:
            self.eraser_message_box = InformationMessageBox("Please do segmentation first!")
            self.eraser_message_box.show()
        elif self.annotation_flag == 1 or self.annotation_flag == 2:
            pix = QPixmap(get_icon("eraser1.png"))
            cursor = QCursor(pix)
            self.visualize_window.setCursor(cursor)
            current_frame = self.frames[self.visualize.main_sld.value()]
            current_id = self.correction.instances_widget.currentRow()
            if current_id >= 0:
                colormap = pg.ColorMap([0, 1], color=[[0, 0, 0], [255, 255, 255]])
                self.visualize.main_frame.setColorMap(colormap)
                self.visualize.main_frame.setImage(current_frame.raw_img[:, :, 0])
                self.visualize.main_frame.getImageItem().mouseDragEvent = self.eraser_drag1
            else:
                self.eraser_message_box = InformationMessageBox("Please add a cell first!")
                self.eraser_message_box.show()
        elif self.annotation_flag == 0:
            pix = QPixmap(get_icon("eraser1.png"))
            cursor = QCursor(pix)
            self.visualize_window.setCursor(cursor)
            self.show_flag = 4
            current_frame = self.frames[self.visualize.main_sld.value()]
            current_id = self.correction.instances_widget.currentRow()
            if current_id >= 0:
                current_color = current_frame.instances[self.widget_list[current_id]].color
                current_coords = current_frame.instances[self.widget_list[current_id]].coords
                x = current_coords[0]
                y = current_coords[1]
                for i in range(len(x)):
                    my_roi = pg.ROI(pos=(y[i] - 0.5, x[i] - 0.5),
                                    pen=QPen(QColor(current_color[0], current_color[1], current_color[2])),
                                    movable=False)
                    self.color_roi.append(my_roi)
                    self.visualize.main_frame.addItem(my_roi)

                colormap = pg.ColorMap([0, 1], color=[[0, 0, 0], [255, 255, 255]])
                self.visualize.main_frame.setColorMap(colormap)
                self.visualize.main_frame.setImage(current_frame.raw_img[:, :, 0])
                self.visualize.main_frame.getImageItem().mouseDragEvent = self.eraser_drag1
            else:
                self.eraser_message_box = InformationMessageBox("Please add a cell first!")
                self.eraser_message_box.show()

    def brush_drag1(self, event):
        event.accept()
        pos = event.pos()
        ssv = int(self.correction.size_editor.text())

        current_frame = self.frames[self.visualize.main_sld.value()]
        current_id    = self.correction.instances_widget.currentRow()
        current_label = current_frame.instances[self.widget_list[current_id]].label_id
        current_name  = current_frame.instances[self.widget_list[current_id]].name
        current_color = current_frame.instances[self.widget_list[current_id]].color
        self.changed_instances.add_update_ins(current_id, current_label, current_name, current_color)

        temp = self.frames[self.visualize.main_sld.value()].label_img
        if self.annotation_flag == 2:
            temp[int(pos[1] - ssv + 1): int(pos[1]) + ssv, int(pos[0]) - ssv + 1: int(pos[0]) + ssv] = current_label
        else:
            temp[int(pos[1] - ssv + 1): int(pos[1]) + ssv, int(pos[0]) - ssv + 1: int(pos[0]) + ssv, 0] = current_label
        roi_my = pg.ROI(pos=(int(pos[0])-0.5, int(pos[1])-0.5), size=(ssv, ssv),
                        pen=QPen(QColor(current_color[0], current_color[1], current_color[2])),
                        movable=False)
        if self.annotation_flag == 2:
            self.frames_roi[self.visualize.main_sld.value()].append(roi_my)
        self.color_roi.append(roi_my)
        self.visualize.main_frame.addItem(roi_my)
        self.frames[self.visualize.main_sld.value()].label_img = temp
        # self.visualize.main_frame.imageItem.updateImage(self.frames[self.visualize.main_sld.value()].label_img[:, :, 0].T)

    def eraser_drag1(self, event):
        event.accept()
        pos = event.pos()
        ssv = int(self.correction.size_editor.text())

        current_frame = self.frames[self.visualize.main_sld.value()]
        current_id = self.correction.instances_widget.currentRow()
        current_label = current_frame.instances[self.widget_list[current_id]].label_id
        current_name = current_frame.instances[self.widget_list[current_id]].name
        current_color = current_frame.instances[self.widget_list[current_id]].color

        self.changed_instances.add_update_ins(current_id, current_label, current_name, current_color)

        temp = self.frames[self.visualize.main_sld.value()].label_img
        if self.annotation_flag == 2:
            temp[int(pos[1] - ssv + 1): int(pos[1]) + ssv, int(pos[0]) - ssv + 1: int(pos[0]) + ssv] = 0
        else:
            temp[int(pos[1] - ssv + 1): int(pos[1]) + ssv, int(pos[0]) - ssv + 1: int(pos[0]) + ssv, 0] = 0
        # temp[int(pos[1]) - ssv + 1: int(pos[1]) + ssv, int(pos[0]) - ssv + 1: int(pos[0]) + ssv, 0] = 0
        self.frames[self.visualize.main_sld.value()].label_img = temp

        del_c_roi = []
        if self.annotation_flag == 2:
            for c_roi1 in self.frames_roi[self.visualize.main_sld.value()]:
                last_pos = c_roi1.state['pos']
                if pos[0] - ssv + 1 <= last_pos[0] <= pos[0] + ssv - 1 and \
                        pos[1]-ssv+1 <= last_pos[1] <= pos[1] + ssv - 1:
                    self.visualize.main_frame.removeItem(c_roi1)
                    del_c_roi.append(c_roi1)
                    self.visualize.main_frame.imageItem.updateImage()
            for c_roi1 in del_c_roi:
                self.frames_roi[self.visualize.main_sld.value()].remove(c_roi1)
        else:
            for c_roi in self.color_roi:
                last_pos = c_roi.state['pos']
                if pos[0] - ssv + 1 <= last_pos[0] <= pos[0] + ssv - 1 and \
                        pos[1] - ssv + 1 <= last_pos[1] <= pos[1] + ssv - 1:
                    self.color_roi.remove(c_roi)
                    del_c_roi.append(c_roi)
            for c_roi in del_c_roi:
                self.visualize.main_frame.removeItem(c_roi)


        if self.show_flag == 3:
            pass
        else:
            pass

    def confirm_fnc(self):
        if len(self.frames) == 0:
            self.confirm_message_box = InformationMessageBox("Please load images first!")
            self.confirm_message_box.show()
        elif not len(self.frames) == 0 and self.frames[0].label_img is None and self.annotation_flag == 0:
            self.confirm_message_box = InformationMessageBox("Please do segmentation first!")
            self.confirm_message_box.show()
        elif self.annotation_flag == 0 or self.annotation_flag == 1:
            current_frame = self.frames[self.visualize.main_sld.value()]
            self.show_flag = 3
            self.visualize.main_frame.clear()
            for i in range(len(self.color_roi)):
                self.visualize.main_frame.removeItem(self.color_roi[i])
            self.color_roi = []
            self.visualize_window.setCursor(Qt.ArrowCursor)
            self.visualize.main_frame.setImage(current_frame.raw_color_img)
            self.frames[self.visualize.main_sld.value()].update_labling(self.changed_instances.update_ins_id,
                                                                        self.changed_instances.update_ins_label,
                                                                        self.changed_instances.update_ins_name,
                                                                        self.changed_instances.update_ins_color,
                                                                        self.changed_instances.delete_ins_id,
                                                                        self.changed_instances.delete_ins_label)
            self.annotation_flag = 0
            self.instance_widget_update_fnc()
            self.instances_widget_fnc()
        elif self.annotation_flag == 2:
            current_frame = self.frames[self.visualize.main_sld.value()]
            colormap = pg.ColorMap(np.linspace(0, 1, current_frame.label_n + 1),
                                   color=current_frame.color_map)
            self.correction.assist_frame.setColorMap(colormap)
            self.correction.assist_frame.setImage(current_frame.label_img)
            self.frames[self.visualize.main_sld.value()].add_labeling(self.changed_instances.update_ins_id,
                                                                      self.changed_instances.update_ins_label,
                                                                      self.changed_instances.update_ins_name,
                                                                      self.changed_instances.update_ins_color)
            self.instance_widget_update_fnc()
            self.instances_widget_fnc()

    def drag_fnc(self):
        if len(self.frames) == 0:
            self.drag_message_box = InformationMessageBox("Please load images first!")
            self.drag_message_box.show()
        elif not len(self.frames) == 0 and self.frames[0].label_img is None and self.annotation_flag == 0:
            self.drag_message_box = InformationMessageBox("Please do segmentation first!")
            self.drag_message_box.show()
        elif self.annotation_flag == 1 or self.annotation_flag == 2 or self.annotation_flag == 0:
            pix = QPixmap(get_icon("hand1.png"))
            # pix = pix.scaled(20, 20, Qt.KeepAspectRatio)
            cursor = QCursor(pix)
            self.visualize_window.setCursor(cursor)
            self.visualize.main_frame.getImageItem().mouseDragEvent = self.drag_begin

    def drag_begin(self, event):
        event.ignore()

    '''
    def switch_show_fnc(self):
        self.show_flag *= -1
        if self.show_flag == 1:
            self.visualize.main_frame.setImage(self.frames[self.visualize.main_sld.value()].binary_mask[:, :, 0])
            self.correction.assist_frame.setImage(self.frames[self.visualize.main_sld.value()].raw_img[:, :, 0])
        else:
            self.visualize.main_frame.setImage(self.frames[self.visualize.main_sld.value()].raw_img[:, :, 0])
            self.correction.assist_frame.setImage(self.frames[self.visualize.main_sld.value()].binary_mask[:, :, 0])
    '''

    def size_editor_fnc(self):
        if int(self.correction.size_editor.text()) > 10:
            self.correction.size_editor.setText(str(10))
        elif int(self.correction.size_editor.text()) < 1:
            self.correction.size_editor.setText(str(1))

    def size_left_fnc(self):
        v = int(self.correction.size_editor.text())
        if v > 1:
            self.correction.size_editor.setText(str(v - 1))

    def size_right_fnc(self):
        v = int(self.correction.size_editor.text())
        if v < 11:
            self.correction.size_editor.setText(str(v + 1))

    def add_instance_fnc(self):

        if self.annotation_flag == 0:
            self.instances_setting = InstanceSettings(self.my_colors, self.frames[self.visualize.main_sld.value()].raw_img,
                                                      self.frames[self.visualize.main_sld.value()].label_max+1,
                                                      self.frames[self.visualize.main_sld.value()].frame_id)
            self.instances_setting.confirm_button.clicked.connect(self.add_instance_main)
            self.instances_setting.show()
            self.annotation_flag = 1
        elif self.annotation_flag == 2:
            self.instances_setting = InstanceSettings(self.my_colors,
                                                      self.frames[self.visualize.main_sld.value()].raw_img,
                                                      self.frames[self.visualize.main_sld.value()].label_max + 1,
                                                      self.frames[self.visualize.main_sld.value()].frame_id)
            self.instances_setting.confirm_button.clicked.connect(self.add_instance_main)
            self.instances_setting.show()
        else:
            self.add_instance_message_box = InformationMessageBox("Please load images or segment first!")
            self.add_instance_message_box.show()

    def add_instance_main(self):
        pix = QPixmap(30, 30)
        pix.fill(QColor(self.instances_setting.color[0],
                        self.instances_setting.color[1],
                        self.instances_setting.color[2]))
        contents_show = "Name: " + self.instances_setting.name + "\n" + \
                        " Centroid: NAF"
        item = QListWidgetItem(contents_show, self.correction.instances_widget)
        item.setIcon(QIcon(pix))
        item.setSizeHint(QSize(35, 35))
        item.setTextAlignment(Qt.AlignCenter)

        self.correction.instances_widget.addItem(item)
        self.widget_list.append(self.instances_setting.ins.label_id)
        self.frames[self.visualize.main_sld.value()].add_instance(self.instances_setting.ins.label_id,
                                                                  self.instances_setting.ins)
        self.correction.instances_widget.setCurrentRow(len(self.widget_list) - 1)

        if self.tools.segment.segment_tools.currentIndex() == 3:
            self.frames[self.visualize.main_sld.value()].update_labling([self.correction.instances_widget.currentRow()],
                                                                        [self.instances_setting.ins.label_id],
                                                                        [self.instances_setting.ins.name],
                                                                        [self.instances_setting.ins.color])

            self.instance_widget_update_fnc()
            self.instances_widget_fnc()
            self.visualize.main_frame.removeItem(self.grabcut_roi)
            if self.bboxroi is not None:
                self.visualize.main_frame.removeItem(self.bboxroi)

            # self.visualize.main_frame.setImage(self.frames[self.visualize.main_sld.value()].raw_color_img)
        current_frame = self.frames[self.visualize.main_sld.value()]
        # colormap = pg.ColorMap([0, 1], color=[[0, 0, 0], [255, 255, 255]])
        # self.visualize.main_frame.setColorMap(colormap)
        self.visualize.main_frame.setImage(current_frame.raw_color_img)

    def instances_widget_fnc(self):
        self.visualize_window.setCursor(Qt.ArrowCursor)
        self.visualize.main_frame.getImageItem().mouseDragEvent = self.drag_begin
        if self.bboxroi is not None:
            self.visualize.main_frame.removeItem(self.bboxroi)

        bbox = self.frames[self.visualize.main_sld.value()].instances[
            self.widget_list[self.correction.instances_widget.currentRow()]].bbox
        if bbox is not None:
            self.bboxroi = pg.RectROI([bbox[0], bbox[1]], [bbox[2]-bbox[0], bbox[3]-bbox[1]])
            self.visualize.main_frame.addItem(self.bboxroi)

    def instances_widget_fnc_double(self):
        self.cell_window = CellAttributeWindow(self.frames[self.visualize.main_sld.value()].instances[
                                              self.widget_list[self.correction.instances_widget.currentRow()]])
        self.cell_window.show()

    def instance_delete_fnc(self):
        current_row = self.correction.instances_widget.currentRow()

        if self.annotation_flag == 2 and current_row >= 0:
            current_frame = self.frames[self.visualize.main_sld.value()]
            current_color = current_frame.instances[self.widget_list[current_row]].color
            for c_roi in self.frames_roi[self.visualize.main_sld.value()]:
                c_roi_color = c_roi.pen.color()
                c_roi_color1 = c_roi_color.getRgb()
                if c_roi_color1[0] == current_color[0] and c_roi_color1[1] == current_color[1] and \
                        c_roi_color1[2] == current_color[2]:
                    self.visualize.main_frame.removeItem(c_roi)

            ins_id = self.widget_list[current_row]
            self.widget_list.remove(ins_id)

            delete_ins_label = []
            delete_ins_id = []
            delete_ins_id.append(current_row)
            delete_ins_label.append(ins_id)
            self.frames[self.visualize.main_sld.value()].update_labling([], [], [], delete_ins_id=delete_ins_id,
                                                                        delete_ins_label=delete_ins_label,
                                                                        annotation_flag=2)
            self.correction.instances_widget.takeItem(current_row)
            if not np.max(self.frames[self.visualize.main_sld.value()].label_img) == \
                    np.min(self.frames[self.visualize.main_sld.value()].label_img):
                colormap = pg.ColorMap(np.linspace(0, 1, self.frames[self.visualize.main_sld.value()].label_n + 1),
                                       color=self.frames[self.visualize.main_sld.value()].color_map)
                self.correction.assist_frame.setColorMap(colormap)
                self.correction.assist_frame.setImage(self.frames[self.visualize.main_sld.value()].label_img,
                                                      autoHistogramRange=False)
            else:
                self.correction.assist_frame.setImage(self.frames[self.visualize.main_sld.value()].raw_img,
                                                      autoHistogramRange=False)

        elif self.annotation_flag == 0 and current_row >= 0:
            current_frame = self.frames[self.visualize.main_sld.value()]
            current_color = current_frame.instances[self.widget_list[current_row]].color
            ins_id = self.widget_list[current_row]
            self.widget_list.remove(ins_id)

            delete_ins_label = []
            delete_ins_id    = []
            delete_ins_id.append(current_row)
            delete_ins_label.append(ins_id)
            self.frames[self.visualize.main_sld.value()].update_labling([], [], [], delete_ins_id=delete_ins_id,
                                                                        delete_ins_label=delete_ins_label)
            self.correction.instances_widget.takeItem(current_row)

            current_frame = self.frames[self.visualize.main_sld.value()]
            self.visualize.main_frame.imageItem.updateImage(current_frame.raw_color_img)
        else:
            self.delete_instance_message_box = InformationMessageBox("Please select a cell first!")
            self.delete_instance_message_box.show()

    def center(self):
        qr = self.frameGeometry()
        self.desktop = QApplication.desktop()
        self.screenRect = self.desktop.screenGeometry()
        qr.moveCenter(self.screenRect.center())
        self.move(qr.topLeft())

    def change_show(self):
        if not self.left_tools.isVisible():
            self.left_tools.setVisible(True)

    # TODO
    def left_navigation_fnc(self):
        if self.last_index == self.left_navigation.currentRow() and self.left_tools.isVisible():
            self.left_tools.setVisible(False)
            self.last_item = self.left_navigation.item(self.last_index)
            self.last_item.setBackground(QColor("#414141"))
        elif self.last_index == self.left_navigation.currentRow() and (not self.left_tools.isVisible()):
            self.left_tools.setVisible(True)
            self.current_item = self.left_navigation.currentItem()
            self.current_item.setBackground(QColor("#323232"))
        elif (not self.last_index == self.left_navigation.currentRow()) and self.left_tools.isVisible():
            self.left_tools.setCurrentIndex(self.left_navigation.currentRow())
            self.last_item = self.left_navigation.item(self.last_index)
            self.last_item.setBackground(QColor("#414141"))
            self.current_item = self.left_navigation.currentItem()
            self.current_item.setBackground(QColor("#323232"))

            self.last_index = self.left_navigation.currentRow()
        else:
            self.left_tools.setCurrentIndex(self.left_navigation.currentRow())
            self.left_tools.setVisible(True)
            self.last_item = self.left_navigation.item(self.last_index)
            self.last_item.setBackground(QColor("#414141"))
            self.current_item = self.left_navigation.currentItem()
            self.current_item.setBackground(QColor("#323232"))

            self.last_index = self.left_navigation.currentRow()

    # algorithm API
    def normalize(self, name):
        if not len(self.frames) == 0:
            normalizer = get_normalizer(name=name)
            for key in self.frames.keys():
                self.frames[key].norm_img = normalizer(self.frames[key].raw_img)
            self.visualize.main_frame.setImage(self.frames[self.visualize.main_sld.value()].norm_img)
        else:
            self.normalize_message_box = InformationMessageBox("Please load images first!")
            self.normalize_message_box.show()

    def segment(self, name, **kwargs):
        if (not len(self.frames) == 0 and self.frames[0].label_img is None) or \
                self.tools.segment.segment_tools.currentIndex() == 3:
            self.segment_message_box1_fnc(name, **kwargs)
        elif not len(self.frames) == 0 and self.frames[0].label_img is not None:
            self.segment_message_box1 = QuestionMessageBox("Are you sure to re-segment the images?")
            self.segment_message_box1.yes_button.clicked.connect(lambda: self.segment_message_box1_fnc(name, **kwargs))
            self.segment_message_box1.cancel_button.clicked.connect(self.segment_message_box1.close_fnc)
            self.segment_message_box1.show()
        else:
            self.segment_message_box2 = InformationMessageBox("Please load images first!")
            self.segment_message_box2.show()

    def segment_message_box1_fnc(self, name, **kwargs):

        if not self.tools.segment.segment_tools.currentIndex() == 3:
            self.status.progressbar.setMaximum(self.frames_num)
            self.status.progressbar.setVisible(True)
            if name == 'unet':
                self.status.work_info_label.setText("Loading selected model...")
            t = 1
            segmenter = get_segmenter(name=name, **kwargs)
            self.status.work_info_label.setText("Segmenting...")
            for key in self.frames.keys():
                label_img = segmenter(self.frames[key].norm_img)
                self.frames[key].label_img = label_img

                self.status.update_progressbar(t)
                t = t + 1
            self.status.progressbar.setVisible(False)
            self.status.work_info_label.setText("")
            sv = self.visualize.main_sld.value()

            self.visualize.main_frame.setImage(np.sign(self.frames[sv].binary_mask[:, :, 0]))
            self.show_flag = 2
            self.annotation_flag = 0

            self.get_label_fnc()
        else:
            segmenter = get_segmenter(name=name, **kwargs)
            self.status.work_info_label.setText("Segmenting...")
            if self.frames[self.visualize.main_sld.value()].label_img is None:
                label_img = segmenter(img=self.frames[self.visualize.main_sld.value()].norm_img,
                                      rect=[int(self.grabcut_roi.state['pos'][0]),
                                            int(self.grabcut_roi.state['pos'][1]),
                                            int(self.grabcut_roi.state['pos'][0]+self.grabcut_roi.state['size'][0]),
                                            int(self.grabcut_roi.state['pos'][1]+self.grabcut_roi.state['size'][1])],
                                      obj_idx=1)
            else:
                label_img = segmenter(img=self.frames[self.visualize.main_sld.value()].raw_img,
                                      rect=[int(self.grabcut_roi.state['pos'][0]),
                                            int(self.grabcut_roi.state['pos'][1]),
                                            int(self.grabcut_roi.state['pos'][0] + self.grabcut_roi.state['size'][0]),
                                            int(self.grabcut_roi.state['pos'][1] + self.grabcut_roi.state['size'][1])],
                                      obj_idx=np.max(self.frames[self.visualize.main_sld.value()].label_img) + 1)

            if label_img.max() == 0:
                self.segment_message_box3 = InformationMessageBox("Segmentation failed!")
                self.segment_message_box3.show()
            else:
                if self.frames[self.visualize.main_sld.value()].label_img is None:
                    self.frames[self.visualize.main_sld.value()].label_img = label_img
                else:
                    self.frames[self.visualize.main_sld.value()].label_img += label_img

                self.instances_setting = InstanceSettings(self.my_colors,
                                                          self.frames[self.visualize.main_sld.value()].raw_img,
                                                          self.frames[self.visualize.main_sld.value()].label_max,
                                                          self.frames[self.visualize.main_sld.value()].frame_id)
                self.instances_setting.confirm_button.clicked.connect(self.add_instance_main)
                self.instances_setting.show()




            # self.visualize.main_frame.setImage(np.sign(label_img[:, :, 0]))

    def track(self, name, **kwargs):
        if not len(self.frames) == 0 and self.frames[0].label_img is None:
            self.track_message_box1 = InformationMessageBox("Please do segment first!")
            self.track_message_box1.show()
        elif not len(self.frames) == 0 and self.frames[0].label_img is not None:
            self.status_fnc()
            tracker = get_tracker(name=name, **kwargs)
            self.frames = tracker(self.frames)
            self.get_label_fnc1()
            self.status.work_info_label.setText("")
        else:
            self.track_message_box2 = InformationMessageBox("Please load images first!")
            self.track_message_box2.show()

    def status_fnc(self):
        QApplication.processEvents()
        self.status.work_info_label.setText("Tracking...")

    def write(self, **kwargs):
        if not len(self.frames) == 0 and self.frames[0].label_img is None:
            self.write_message_box1 = InformationMessageBox("Please do segment first!")
            self.write_message_box1.show()
        elif not len(self.frames) == 0 and self.frames[0].label_img is not None:
            self.status.work_info_label.setText("Outputing...")
            QApplication.processEvents()
            # self.status.progressbar.setVisible(True)
            writer = get_writer(**kwargs)
            writer(self.frames)
            # self.status.progressbar.setVisible(True)
            self.status.work_info_label.setText("")
        else:
            self.write_message_box2 = InformationMessageBox("Please load images first!")
            self.write_message_box2.show()

    def my_update(self):
        self.visualize.main_frame.clear()

        colormap = pg.ColorMap(np.linspace(0, 1, self.frames[self.main_sld.value()].label_n+1),
                               color=self.frames[self.main_sld.value()].color_map)
        self.visualize.main_frame.setImage(self.frames[self.main_sld.value()].label_img[:, :, 0])
        self.visualize.main_frame.setColorMap(colormap)

    def get_label_fnc(self):
        """get instances"""
        self.status.progressbar.setMaximum(self.frames_num)
        self.status.progressbar.setVisible(True)
        self.status.work_info_label.setText("Labeling...")

        t = 1
        for key in self.frames.keys():
            self.status.update_progressbar(t)
            self.frames[key].auto_labling(self.my_colors)
            t = t+1
        self.status.progressbar.setVisible(False)
        self.status.work_info_label.setText("")

        current_frame = self.frames[self.visualize.main_sld.value()]

        '''
        colormap = pg.ColorMap(np.linspace(0, 1, current_frame.label_n+1),
                               color=self.frames[self.visualize.main_sld.value()].color_map)
        '''
        self.visualize.main_frame.setImage(current_frame.raw_color_img)
        # self.visualize.main_frame.setColorMap(colormap)
        self.widget_list = []
        for key in self.frames[self.visualize.main_sld.value()].instances.keys():
            pix = QPixmap(30, 30)
            pix.fill(QColor(current_frame.instances[key].color[0],
                            current_frame.instances[key].color[1],
                            current_frame.instances[key].color[2]))
            contents_show = "Name: " + self.frames[self.visualize.main_sld.value()].instances[key].name + "\n" +\
                            " Centroid: " + str(self.frames[self.visualize.main_sld.value()].instances[key].centroid)
            item = QListWidgetItem(contents_show, self.correction.instances_widget)
            item.setIcon(QIcon(pix))
            item.setSizeHint(QSize(35, 35))
            # item.setSizeHint(Qsize(100, 100))
            self.widget_list.append(self.frames[self.visualize.main_sld.value()].instances[key].label_id)
            item.setTextAlignment(Qt.AlignCenter)
            self.correction.instances_widget.addItem(item)
        self.show_flag = 3
        # self.status_signal[1] = 1

    def get_label_fnc1(self):
        """get instances"""
        self.status.progressbar.setMaximum(self.frames_num)
        self.status.progressbar.setVisible(True)
        self.status.work_info_label.setText("Labeling...    ")

        t = 1
        color_map_dict = None
        for key in self.frames.keys():
            self.status.update_progressbar(t)
            self.frames[key].auto_labeling1(self.my_colors, color_map_dict)
            t = t+1
            color_map_dict = self.frames[key].color_map_dict
            # print(key, color_map_dict)
        self.status.progressbar.setVisible(False)
        self.status.work_info_label.setText("")

        current_frame = self.frames[self.visualize.main_sld.value()]

        '''
        colormap = pg.ColorMap(np.linspace(0, 1, current_frame.label_n+1),
                               color=self.frames[self.visualize.main_sld.value()].color_map)
        '''
        self.visualize.main_frame.setImage(current_frame.raw_color_img)
        # self.visualize.main_frame.setImage(self.frames[self.visualize.main_sld.value()].label_img[:, :, 0])
        # self.visualize.main_frame.setColorMap(colormap)
        self.widget_list = []
        self.correction.instances_widget.clear()
        for key in self.frames[self.visualize.main_sld.value()].instances.keys():
            pix = QPixmap(30, 30)
            pix.fill(QColor(current_frame.instances[key].color[0],
                            current_frame.instances[key].color[1],
                            current_frame.instances[key].color[2]))
            contents_show = "Name: " + self.frames[self.visualize.main_sld.value()].instances[key].name + "\n" + \
                            " Centroid: " + str(self.frames[self.visualize.main_sld.value()].instances[key].centroid)
            item = QListWidgetItem(contents_show, self.correction.instances_widget)
            item.setIcon(QIcon(pix))
            item.setSizeHint(QSize(35, 35))
            # item.setSizeHint(Qsize(100, 100))
            self.widget_list.append(self.frames[self.visualize.main_sld.value()].instances[key].label_id)
            item.setTextAlignment(Qt.AlignCenter)
            self.correction.instances_widget.addItem(item)
        self.show_flag = 3
        # self.status_signal[1] = 1

    '''
    def load_data_first_message(self):
        self.info_message_box = InformationMessageBox("Please load images first!")

    def segment_first_message(self):
        self.segment_message_box = InformationMessageBox("Please do segmentation first!")

    def select_result_path_message(self):
        self.result_path_message_box = InformationMessageBox("Please select the result path first!")
    '''


def run():
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())
