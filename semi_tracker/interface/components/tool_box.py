import os.path as osp
import random
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import QSize, Qt
from PyQt5 import QtCore, QtWidgets
from PyQt5 import QtGui
import pyqtgraph as pg
from .frame import Instance
from ..utils import get_icon
import cv2


class ToolBox(QWidget):
    def __init__(self):
        super().__init__()

        self.toolbox = QWidget()

        self.setup_ui()

    def setup_ui(self):
        self.init_tool1()
        self.init_tool2()

        tool_layout = QVBoxLayout()

        tool1_button = QPushButton()
        tool1_button.setText("tool1")
        tool1_button.setStyleSheet("background: #353535;"
                                   "font-family: Verdana;")
        tool1_button.setFixedHeight(30)
        tool1_button.clicked.connect(self.tool1_button_fnc)
        tool2_button = QPushButton()
        tool2_button.setText("tool2")
        tool2_button.setStyleSheet("background: #505050;"
                                   "font-family: Verdana;")
        tool2_button.setFixedHeight(30)
        tool2_button.clicked.connect(self.tool2_button_fnc)

        tool_layout.addWidget(tool1_button)
        tool_layout.addWidget(self.tool1)
        tool_layout.addWidget(tool2_button)
        tool_layout.addWidget(self.tool2)
        tool_layout.setAlignment(Qt.AlignTop)
        tool_layout.setContentsMargins(0, 0, 0, 0)
        self.toolbox.setLayout(tool_layout)

        self.tool1_button = tool1_button
        self.tool2_button = tool2_button
        self.tool_layout = tool_layout


    def init_tool1(self):
        tool1 = QWidget()
        tool1.setStyleSheet("background: #707070;")
        tool1_layout = QHBoxLayout()


        tool1_label = QLabel()
        tool1_label.setText("tool1")

        tool1_layout.addWidget(tool1_label)

        tool1.setLayout(tool1_layout)

        self.tool1 = tool1

    def init_tool2(self):
        tool2 = QWidget()
        tool2.setStyleSheet("background: #707070;")
        tool2_layout = QHBoxLayout()

        tool2_label = QLabel()
        tool2_label.setText("tool2")

        tool2_layout.addWidget(tool2_label)

        tool2.setLayout(tool2_layout)

        self.tool2 = tool2

    def tool1_button_fnc(self):
        if self.tool1.isVisible():
            self.tool1.setVisible(False)

        else:
            self.tool1.setVisible(True)

    def tool2_button_fnc(self):
        if self.tool2.isVisible():
            self.tool2.setVisible(False)
        else:
            self.tool2.setVisible(True)
