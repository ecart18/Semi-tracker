# -*- coding: UTF-8 -*-

import os.path as osp
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from .file_tree import FileTree
from .left_tools_segment import SegmentTools
from .left_tools_output import OutputTools
from .left_tools_track import TrackTools
from .left_tools_annotation import AnnotationTools
from .left_tools_normalization import NormalizeTools
from ..utils import get_icon, left_tools_stylesheet


class LeftTools(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.left_tools = QStackedWidget()
        self.left_tools.setStyleSheet("background: #323232;"
                                      "border: 0px;")
        self.left_tools.setFixedWidth(280)

        self.project_path = "./output/untitled"

        self.setup_ui()

    def setup_ui(self):

        # self.left_tools.clicked.connect(self.click_event)

        self.init_file_tree()
        self.init_img_preprocess()
        self.init_algorithms()
        self.init_annotation()

    def init_file_tree(self):
        output_path = osp.dirname(self.project_path)
        # print(output_path)
        self.file_system = QWidget()
        self.file_system_layout = QVBoxLayout(self.file_system)
        self.file_system_layout.setContentsMargins(0, 0, 0, 0)
        self.file_tree_widget = FileTree()
        self.file_system_layout.addWidget(self.file_tree_widget)
        self.left_tools.addWidget(self.file_system)

    def init_img_preprocess(self):
        self.left_tools.addWidget(QLabel("Image preprocess"))

    def init_algorithms(self):
        self.normlize  = NormalizeTools()
        self.segment   = SegmentTools()
        self.track     = TrackTools()
        self.output    = OutputTools()

        self.left_tools_stylesheet = left_tools_stylesheet

        self.main_algorithm = QToolBox()
        self.main_algorithm.setStyleSheet(self.left_tools_stylesheet)
        self.main_algorithm.addItem(self.normlize.normalize_tools, "Normalization")
        self.main_algorithm.addItem(self.segment.segment_tools, "Segmentation")
        self.main_algorithm.addItem(self.track.track_tools, "Track")
        self.main_algorithm.addItem(self.output.output_tools, "Output")
        self.main_algorithm.setItemIcon(0, QIcon(get_icon("Arrow_down.png")))
        self.main_algorithm.setItemIcon(1, QIcon(get_icon("Arrow_right.png")))
        self.main_algorithm.setItemIcon(2, QIcon(get_icon("Arrow_right.png")))
        self.main_algorithm.setItemIcon(3, QIcon(get_icon("Arrow_right.png")))
        self.main_algorithm.layout().setSpacing(2)
        # main_algorithm_layout = QVBoxLayout(self.main_algorithm)
        # main_algorithm_layout.addWidget(self.segment.segment_tools)
        # main_algorithm_layout.addWidget(self.track.track_tools)
        # main_algorithm_layout.addWidget(self.output.output_tools)
        self.left_tools.addWidget(self.main_algorithm)

    def init_annotation(self):
        self.annotation = AnnotationTools()
        self.left_annotation = self.annotation.annotation_tools

        self.left_tools.addWidget(self.left_annotation)

    def update_file_tree(self, project_path):
        self.project_path = project_path
        self.file_system_layout.removeWidget(self.file_tree_widget)
        self.file_tree_widget = FileTree(self.project_path)
        self.file_system_layout.addWidget(self.file_tree_widget)

