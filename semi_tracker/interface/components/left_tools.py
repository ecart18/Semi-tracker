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
from .tool_box import ToolBox


class LeftTools(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.left_tools = QStackedWidget()
        self.left_tools.setStyleSheet("background: #323232;"
                                      "border: 0px;")
        self.left_tools.setFixedWidth(250)

        self.project_path = "./output/untitled"

        self.setup_ui()

    def setup_ui(self):

        # self.left_tools.clicked.connect(self.click_event)

        self.init_file_tree()
        # self.init_img_preprocess()
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

    def init_algorithms(self):
        self.normlize  = NormalizeTools()
        self.segment   = SegmentTools()
        self.track     = TrackTools()
        self.output    = OutputTools()

        self.left_tools_stylesheet = left_tools_stylesheet

        self.main_algorithm = QWidget()
        #self.main_algorithm.setFixedHeight(200)
        self.main_algorithm.setStyleSheet("background: #323232")

        self.main_algorithm_layout = QVBoxLayout()
        self.main_algorithm_layout.setAlignment(Qt.AlignTop)
        self.main_algorithm_layout.setContentsMargins(0, 0, 0, 0)
        self.main_algorithm_layout.setSpacing(1)

        self.normlize_button = QPushButton()
        self.normlize_button.setStyleSheet("background: #454545;"
                                                     "font-family: Verdana;"
                                           "font-size: 15px;"
                                                     "border: 0px;"
                                                     "text-align:left;")
        self.normlize_button.setFixedHeight(25)
        self.normlize_button.setText("Normalization")
        self.normlize_button.clicked.connect(self.normlize_button_fnc)
        self.normlize_button.setIcon(QIcon(get_icon("Arrow_right.png")))
        # self.equalize_hist_tool_button.setFlat(True)

        self.segment_button = QPushButton()
        self.segment_button.setStyleSheet("background: #454545;"
                                               "font-family: Verdana;"
                                          "font-size: 15px;"
                                               "border: 0px;"
                                               "text-align:left;")
        self.segment_button.setFixedHeight(25)
        self.segment_button.setText("Segmentation")
        self.segment_button.clicked.connect(self.segment_button_fnc)
        self.segment_button.setIcon(QIcon(get_icon("Arrow_right.png")))
        # self.min_max_tool_button.setFlat(True)

        self.track_button = QPushButton()
        self.track_button.setStyleSheet("background: #454545;"
                                                     "font-family: Verdana;"
                                        "font-size: 15px;"
                                                     "border: 0px;"
                                                     "text-align:left;")
        self.track_button.setFixedHeight(25)
        self.track_button.setText("Track")
        self.track_button.clicked.connect(self.track_button_fnc)
        self.track_button.setIcon(QIcon(get_icon("Arrow_right.png")))
        # self.retinex_MSRCP_tool_button.setFlat(True)

        self.output_button = QPushButton()
        self.output_button.setStyleSheet("background: #454545;"
                                                     "font-family: Verdana;"
                                         "font-size: 15px;"
                                                     "border: 0px;"
                                                     "text-align:left;")
        self.output_button.setFixedHeight(25)
        self.output_button.setText("Output")
        self.output_button.clicked.connect(self.output_button_fnc)
        self.output_button.setIcon(QIcon(get_icon("Arrow_right.png")))
        # self.retinex_MSRCR_tool_button.setFlat(True)

        self.main_algorithm_layout.addWidget(self.normlize_button)
        self.main_algorithm_layout.addWidget(self.normlize.normalize_tools)
        self.normlize.normalize_tools.setVisible(False)
        self.main_algorithm_layout.addWidget(self.segment_button)
        self.main_algorithm_layout.addWidget(self.segment.segment_tools)
        self.segment.segment_tools.setVisible(False)
        self.main_algorithm_layout.addWidget(self.track_button)
        self.main_algorithm_layout.addWidget(self.track.track_tools)
        self.track.track_tools.setVisible(False)
        self.main_algorithm_layout.addWidget(self.output_button)
        self.main_algorithm_layout.addWidget(self.output.output_tools)
        self.output.output_tools.setVisible(False)

        self.main_algorithm.setLayout(self.main_algorithm_layout)


        """
        self.main_algorithm.addItem(self.normlize.normalize_tools, "Normalization")
        self.main_algorithm.addItem(self.segment.segment_tools, "Segmentation")
        self.main_algorithm.addItem(self.track.track_tools, "Track")
        self.main_algorithm.addItem(self.output.output_tools, "Output")
        self.main_algorithm.addItem(self.algorithm_widget, "")
        # self.main_algorithm.setItem
        self.main_algorithm.setItemIcon(0, QIcon(get_icon("Arrow_right.png")))
        self.main_algorithm.setItemIcon(1, QIcon(get_icon("Arrow_right.png")))
        self.main_algorithm.setItemIcon(2, QIcon(get_icon("Arrow_right.png")))
        self.main_algorithm.setItemIcon(3, QIcon(get_icon("Arrow_right.png")))
        self.main_algorithm.layout().setSpacing(2)
        self.main_algorithm.setCurrentIndex(4)
        # self.main_algorithm.setStyle()
        # main_algorithm_layout = QVBoxLayout(self.main_algorithm)
        # main_algorithm_layout.addWidget(self.segment.segment_tools)
        # main_algorithm_layout.addWidget(self.track.track_tools)
        # main_algorithm_layout.addWidget(self.output.output_tools)
        """
        self.left_tools.addWidget(self.main_algorithm)

    def init_annotation(self):
        self.annotation = AnnotationTools()
        # self.toolbox = ToolBox()
        # self.toolbox_box = self.toolbox.toolbox
        self.left_annotation = self.annotation.annotation_tools

        self.left_tools.addWidget(self.left_annotation)
        # self.left_tools.addWidget(self.toolbox_box)

    def update_file_tree(self, project_path):
        self.project_path = project_path
        self.file_system_layout.removeWidget(self.file_tree_widget)
        self.file_tree_widget = FileTree(self.project_path)
        self.file_system_layout.addWidget(self.file_tree_widget)

    def normlize_button_fnc(self):
        if self.normlize.normalize_tools.isVisible():
            self.normlize.normalize_tools.setVisible(False)
            self.normlize_button.setIcon(QIcon(get_icon("Arrow_right.png")))
            self.normlize_button.setStyleSheet("background: #454545;"
                                                         "font-family: Verdana;"
                                                         "border: 0px;"
                                                         "text-align:left;"
                                                         "color: #000000")
        else:
            self.normlize.normalize_tools.setVisible(True)
            self.normlize_button.setIcon(QIcon(get_icon("Arrow_down.png")))
            self.normlize_button.setStyleSheet("background: #454545;"
                                                         "font-family: Verdana;"
                                                         "border: 0px;"
                                                         "text-align:left;"
                                                         "color: #FFFFFF")

    def segment_button_fnc(self):
        if self.segment.segment_tools.isVisible():
            self.segment.segment_tools.setVisible(False)
            self.segment_button.setIcon(QIcon(get_icon("Arrow_right.png")))
            self.segment_button.setStyleSheet("background: #454545;"
                                                         "font-family: Verdana;"
                                                         "border: 0px;"
                                                         "text-align:left;"
                                                         "color: #000000")
        else:
            self.segment.segment_tools.setVisible(True)
            self.segment_button.setIcon(QIcon(get_icon("Arrow_down.png")))
            self.segment_button.setStyleSheet("background: #454545;"
                                                         "font-family: Verdana;"
                                                         "border: 0px;"
                                                         "text-align:left;"
                                                         "color: #FFFFFF")

    def track_button_fnc(self):
        if self.track.track_tools.isVisible():
            self.track.track_tools.setVisible(False)
            self.track_button.setIcon(QIcon(get_icon("Arrow_right.png")))
            self.track_button.setStyleSheet("background: #454545;"
                                                         "font-family: Verdana;"
                                                         "border: 0px;"
                                                         "text-align:left;"
                                                         "color: #000000")
        else:
            self.track.track_tools.setVisible(True)
            self.track_button.setIcon(QIcon(get_icon("Arrow_down.png")))
            self.track_button.setStyleSheet("background: #454545;"
                                                         "font-family: Verdana;"
                                                         "border: 0px;"
                                                         "text-align:left;"
                                                         "color: #FFFFFF")

    def output_button_fnc(self):
        if self.output.output_tools.isVisible():
            self.output.output_tools.setVisible(False)
            self.output_button.setIcon(QIcon(get_icon("Arrow_right.png")))
            self.output_button.setStyleSheet("background: #454545;"
                                                         "font-family: Verdana;"
                                                         "border: 0px;"
                                                         "text-align:left;"
                                                         "color: #000000")
        else:
            self.output.output_tools.setVisible(True)
            self.output_button.setIcon(QIcon(get_icon("Arrow_down.png")))
            self.output_button.setStyleSheet("background: #454545;"
                                                         "font-family: Verdana;"
                                                         "border: 0px;"
                                                         "text-align:left;"
                                                         "color: #FFFFFF")

