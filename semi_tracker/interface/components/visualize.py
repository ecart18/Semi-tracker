# -*- coding: UTF-8 -*-

import os.path as osp
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pyqtgraph as pg
from ..utils import get_icon


class VisualizeWindow(QWidget):

    def __init__(self):
        super().__init__()

        self.sld_stylesheet = """
            QSlider:horizontal 
            {
                min-height: 20px;
            }
            QSlider::groove:horizontal {
                height: 1px;
                background: white; 
            }
            QSlider::handle:horizontal {
                width: 12px;
                margin-top: -6px;
                margin-bottom: -6px;
                border-radius: 6px;
                background: qradialgradient(spread:reflect, cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0.7 rgba(210, 210, 210, 255), stop:0.7 rgba(210, 210, 210, 255));
            }
            QSlider::handle:horizontal:hover {
                background: qradialgradient(spread:reflect, cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0.7 rgba(255, 255, 255, 255), stop:0.7 rgba(255, 255, 255, 255));
            }
        """

        self.visualize_window = QWidget()
        self.visualize_window.setStyleSheet("background: #212121;")

        self.setup_ui()

    def setup_ui(self):
        main_frame = pg.ImageView()
        main_frame.ui.roiBtn.hide()
        main_frame.ui.menuBtn.hide()
        main_frame.ui.histogram.hide()
        main_frame.view.mouseClickEvent = self.my_mouse_click_event
        colormap = pg.ColorMap([0, 1], color=[[0, 0, 0], [255, 255, 255]])
        main_frame.setColorMap(colormap)

        # pg.setConfigOption()

        main_left_button = QPushButton()
        main_left_button.setToolTip("Previous frame")
        main_left_button.setIcon(QIcon(get_icon("left1.png")))
        main_left_button.setIconSize(QSize(30, 30))
        main_left_button.setFlat(True)
        main_left_button.setStyleSheet("border: 0px")
        # main_left.clicked.connect(self.main_left_fnc)

        main_sld = QSlider(Qt.Horizontal, self)
        # main_sld.valueChanged.connect(self.sld_update)

        main_sld.setStyleSheet(self.sld_stylesheet)

        main_right_button = QPushButton()
        main_right_button.setToolTip("Next frame")
        main_right_button.setIcon(QIcon((get_icon("right1.png"))))
        main_right_button.setIconSize(QSize(30, 30))
        main_right_button.setFlat(True)
        main_right_button.setStyleSheet("border: 0px")
        # main_right_button.clicked.connect(self.main_right_fnc)

        sld_layout = QHBoxLayout()
        sld_layout.addWidget(main_left_button)
        sld_layout.addWidget(main_sld)
        sld_layout.addWidget(main_right_button)

        self.visualize_window_layout = QVBoxLayout()
        self.visualize_window_layout.addWidget(main_frame)
        self.visualize_window_layout.addLayout(sld_layout)
        self.visualize_window_layout.setContentsMargins(1, 1, 1, 1)
        self.visualize_window.setLayout(self.visualize_window_layout)

        self.main_frame     = main_frame
        self.main_sld       = main_sld
        self.main_left      = main_left_button
        self.main_right     = main_right_button

    def my_mouse_click_event(self, ev):
        if ev.button() == Qt.RightButton:
            ev.ignore()
