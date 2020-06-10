# -*- coding: UTF-8 -*-

import os.path as osp
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class OutputTools(QWidget):

    def __init__(self):
        super().__init__()
        self.output_tools = QWidget()
        self.output_tools.setStyleSheet("background: #323232;"
                                        "border-radius: 10px;")

        self.setup_ui()

    def setup_ui(self):

        output_label = QLabel()
        output_label.setText("Output format:")
        output_label.setStyleSheet("font-family: Verdana;"
                                   "color: white;")

        output_format_select = QComboBox()
        output_format_select.setFixedSize(80, 20)
        output_format_select.setStyleSheet("border: 0px;"
                                           "color: white;"
                                           "font-family: Verdana;"
                                           "background: transparent;")
        output_format_select.addItem("All")
        output_format_select.addItem("Video")
        output_format_select.addItem("Html")
        output_format_select.addItem("Tree")

        output_button = QPushButton()
        output_button.setText("Output")
        output_button.setFixedSize(60, 20)
        output_button.setStyleSheet("background: #454545;"
                                    "color: white;"
                                    "border-radius: 5px;"
                                    "font-family: Verdana;")
        '''
        output_button.clicked.connect(lambda: self.write(write_folder=self.project_path,
                                                         proj_name='Test_name',
                                                         cell_type='SIM'))
        '''
        output_layout1 = QHBoxLayout()
        output_layout2 = QHBoxLayout()
        output_layout_main = QVBoxLayout()
        output_layout1.addWidget(output_label)
        output_layout1.addWidget(output_format_select)
        output_layout2.addWidget(output_button)
        output_layout2.setAlignment(Qt.AlignRight)
        output_layout_main.addLayout(output_layout1)
        output_layout_main.addLayout(output_layout2)
        output_layout_main.setAlignment(Qt.AlignTop)

        self.output_tools.setLayout(output_layout_main)

        self.output_format_select   = output_format_select
        self.output_button          = output_button
