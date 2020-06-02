# -*- coding: UTF-8 -*-

import os.path as osp
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from ..utils import get_icon


class SegmentTools(QWidget):

    def __init__(self):
        super().__init__()

        self.segment_tools_stylesheet = """
            QToolBox 
            {
                background: #282828;
                padding-bottom: 0px;
            }
            QToolBox::tab 
            {
                font-family: Verdana;
                background: #454545;
                border: 0px;
            }
            QToolBoxButton 
            {
                min-height: 20px;
            }
            QToolBox::tab:selected 
            { 
                color: white;
            }
        """
        self.sld_stylesheet = """
            QSlider:horizontal 
            {
                min-height: 20px;
            }
            QSlider::groove:horizontal 
            {
                height: 1px;
                background: white; 
            }
            QSlider::handle:horizontal 
            {
                width: 12px;
                margin-top: -6px;
                margin-bottom: -6px;
                border-radius: 6px;
                background: qradialgradient(spread:reflect, cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0.7 rgba(210, 210, 210, 255), stop:0.7 rgba(210, 210, 210, 255));
            }
            QSlider::handle:horizontal:hover 
            {
                background: qradialgradient(spread:reflect, cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0.7 rgba(255, 255, 255, 255), stop:0.7 rgba(255, 255, 255, 255));
            }
        """
        self.segment_tools = QToolBox()
        self.segment_tools.setStyleSheet(self.segment_tools_stylesheet)
        self.segment_tools.layout().setSpacing(1)

        self.setup_ui()

    def setup_ui(self):
        self.init_seg1()
        self.init_seg2()
        self.init_seg3()
        self.init_seg4()
        self.init_user_defined()

        self.segment_tools.addItem(self.segment_algorithm1, "Threshold")
        self.segment_tools.addItem(self.segment_algorithm2, "U-net")
        self.segment_tools.addItem(self.segment_algorithm3, "WaterShred")
        self.segment_tools.addItem(self.segment_algorithm4, "CrabCut")
        self.segment_tools.addItem(self.segment_algorithm5, "User-defined")

    def init_seg1(self):
        segment_algorithm1 = QWidget()
        segment_algorithm1.setStyleSheet("border: 0px;"
                                         "background: #323232;")

        segment_algorithm1_layout = QVBoxLayout(segment_algorithm1)
        segment_algorithm1_layout1 = QHBoxLayout()
        segment_algorithm1_layout2 = QHBoxLayout()
        segment_algorithm1_layout3 = QHBoxLayout()

        thresh_sld1 = QSlider(Qt.Horizontal)
        thresh_sld1.setMinimum(0)
        thresh_sld1.setMaximum(255)
        thresh_sld1.setValue(129)
        thresh_sld1.setStyleSheet(self.sld_stylesheet)
        thresh_sld1.valueChanged.connect(self.sld2text1)

        thresh_label1 = QLabel()
        thresh_label1.setText("Threshold(0~255)")
        thresh_label1.setAlignment(Qt.AlignCenter)
        thresh_label1.setStyleSheet("font-family: Verdana;"
                                    "color: white;")

        thresh_textline1 = QLineEdit()
        thresh_textline1.setAlignment(Qt.AlignCenter)
        thresh_textline1.setFixedSize(50, 15)
        thresh_textline1.setValidator(QIntValidator())
        thresh_textline1.setText("129")
        thresh_textline1.setStyleSheet("background: #454545;"
                                       "border: 0px;"
                                       "color: white;"
                                       "border-radius: 5px;"
                                       "font-family: Verdana;")
        thresh_textline1.textEdited.connect(self.text2sld1)

        thresh_segment_button = QPushButton()
        thresh_segment_button.setFixedSize(180, 20)
        thresh_segment_button.setText("Threshold")
        thresh_segment_button.setStyleSheet("background: #454545;"
                                            "color: white;"
                                            "border-radius: 5px;"
                                            "font-family: Verdana;")
        segmenter_name = 'binary_thresholding'
        # thresh_segment_button.clicked.connect(lambda: self.segment(segmenter_name, threshold=self.thresh_sld.value()))

        segment_algorithm1_layout1.addWidget(thresh_label1)
        segment_algorithm1_layout1.setAlignment(Qt.AlignLeft)
        segment_algorithm1_layout2.addWidget(thresh_sld1)
        segment_algorithm1_layout2.addWidget(thresh_textline1)
        segment_algorithm1_layout2.setAlignment(Qt.AlignCenter)
        segment_algorithm1_layout3.addWidget(thresh_segment_button)
        segment_algorithm1_layout3.setAlignment(Qt.AlignRight)

        segment_algorithm1_layout.addLayout(segment_algorithm1_layout1)
        segment_algorithm1_layout.addLayout(segment_algorithm1_layout2)
        segment_algorithm1_layout.addLayout(segment_algorithm1_layout3)
        segment_algorithm1_layout.setAlignment(Qt.AlignTop)
        segment_algorithm1_layout.setSpacing(5)
        segment_algorithm1_layout.setContentsMargins(5, 5, 5, 5)

        self.segment_algorithm1     = segment_algorithm1
        self.thresh_sld1            = thresh_sld1
        self.thresh_label1          = thresh_label1
        self.thresh_textline1       = thresh_textline1
        self.thresh_segment_button  = thresh_segment_button

    def init_seg2(self):
        segment_algorithm2 = QWidget()
        segment_algorithm2.setStyleSheet("border: 0px;"
                                         "background: #323232;")

        segment_algorithm2_layout = QVBoxLayout(segment_algorithm2)
        segment_algorithm2_layout1 = QHBoxLayout()
        segment_algorithm2_layout2 = QHBoxLayout()
        segment_algorithm2_layout3 = QHBoxLayout()
        segment_algorithm2_layout4 = QHBoxLayout()

        model_select_label = QLabel()
        model_select_label.setText("Model: ")
        model_select_label.setFixedSize(50, 20)
        model_select_label.setStyleSheet("font-family: Verdana;"
                                         "color: white;")

        model_path_label = QLineEdit()
        model_path_label.setText("Select a model")
        model_path_label.setFixedSize(180, 20)
        model_path_label.setStyleSheet("background: #454545;"
                                       "border: 0px;"
                                       "color: white;"
                                       "border-radius: 5px;"
                                       "font-family: Verdana;")

        model_browse_button = QPushButton()
        model_browse_button.setIcon(QIcon((get_icon("browse.png"))))
        model_browse_button.setIconSize(QSize(15, 15))
        model_browse_button.setFlat(True)
        model_browse_button.setStyleSheet("border: 0px")
        model_browse_button.setFixedSize(20, 20)
        model_browse_button.setStyleSheet("background: #454545;"
                                          "color: white;"
                                          "border-radius: 5px;"
                                          "font-family: Verdana;")
        # model_browse_button.clicked.connect(self.model_select_fnc)

        thresh_sld2 = QSlider(Qt.Horizontal)
        thresh_sld2.setMinimum(0)
        thresh_sld2.setMaximum(10)
        thresh_sld2.setValue(5)
        thresh_sld2.setStyleSheet(self.sld_stylesheet)
        thresh_sld2.valueChanged.connect(self.sld2text2)

        thresh_label2 = QLabel()
        thresh_label2.setText("Threshold(0~1)")
        thresh_label2.setAlignment(Qt.AlignCenter)
        thresh_label2.setStyleSheet("font-family: Verdana;"
                                    "color: white;")

        thresh_textline2 = QLineEdit()
        thresh_textline2.setAlignment(Qt.AlignCenter)
        thresh_textline2.setFixedSize(50, 15)
        # thresh_textline2.setValidator(QtGui.Validator)
        thresh_textline2.setText("0.5")
        thresh_textline2.setStyleSheet("background: #454545;"
                                       "border: 0px;"
                                       "color: white;"
                                       "border-radius: 5px;"
                                       "font-family: Verdana;")
        thresh_textline2.textEdited.connect(self.text2sld2)

        unet_segment_button = QPushButton()
        unet_segment_button.setFixedSize(180, 20)
        unet_segment_button.setText("U-net segment")
        unet_segment_button.setStyleSheet("background: #454545;"
                                          "color: white;"
                                          "border-radius: 5px;"
                                          "font-family: Verdana;")
        # model_path = "C:\\Users\\Administrator\\Desktop\\xsx1\\semi-tracker-develop\\checkpoint\\model_best.pth.tar"
        segmenter_name = 'unet'
        # unet_segment_button.clicked.connect(lambda: self.segment(segmenter_name, threshold=self.thresh_sld.value()))

        segment_algorithm2_layout1.addWidget(model_select_label)
        segment_algorithm2_layout1.addWidget(model_path_label)
        segment_algorithm2_layout1.addWidget(model_browse_button)
        segment_algorithm2_layout1.setSpacing(1)
        segment_algorithm2_layout2.addWidget(thresh_label2)
        segment_algorithm2_layout2.setAlignment(Qt.AlignLeft)
        segment_algorithm2_layout3.addWidget(thresh_sld2)
        segment_algorithm2_layout3.addWidget(thresh_textline2)
        segment_algorithm2_layout3.setAlignment(Qt.AlignCenter)
        segment_algorithm2_layout4.addWidget(unet_segment_button)
        segment_algorithm2_layout4.setAlignment(Qt.AlignRight)

        segment_algorithm2_layout.addLayout(segment_algorithm2_layout1)
        segment_algorithm2_layout.addLayout(segment_algorithm2_layout2)
        segment_algorithm2_layout.addLayout(segment_algorithm2_layout3)
        segment_algorithm2_layout.addLayout(segment_algorithm2_layout4)
        segment_algorithm2_layout.setAlignment(Qt.AlignTop)
        segment_algorithm2_layout.setSpacing(5)
        segment_algorithm2_layout.setContentsMargins(5, 5, 5, 5)

        self.segment_algorithm2     = segment_algorithm2
        self.model_select_label     = model_select_label
        self.model_path_label       = model_path_label
        self.model_browse_button    = model_browse_button
        self.thresh_label2          = thresh_label2
        self.thresh_sld2            = thresh_sld2
        self.thresh_textline2       = thresh_textline2
        self.unet_segment_button    = unet_segment_button

    def init_seg3(self):
        segment_algorithm3 = QWidget()
        segment_algorithm3.setStyleSheet("border: 0px;"
                                         "background: #323232;")

        segment_algorithm3_layout = QVBoxLayout(segment_algorithm3)
        segment_algorithm3_layout1 = QHBoxLayout()
        segment_algorithm3_layout2 = QHBoxLayout()
        segment_algorithm3_layout3 = QHBoxLayout()

        thresh_sld3 = QSlider(Qt.Horizontal)
        thresh_sld3.setMinimum(0)
        thresh_sld3.setMaximum(255)
        thresh_sld3.setValue(129)
        thresh_sld3.setStyleSheet(self.sld_stylesheet)
        thresh_sld3.valueChanged.connect(self.sld2text3)

        thresh_label3 = QLabel()
        thresh_label3.setText("Threshold(0~255)")
        thresh_label3.setAlignment(Qt.AlignCenter)
        thresh_label3.setStyleSheet("font-family: Verdana;"
                                    "color: white;")

        thresh_textline3 = QLineEdit()
        thresh_textline3.setAlignment(Qt.AlignCenter)
        thresh_textline3.setFixedSize(50, 15)
        thresh_textline3.setValidator(QIntValidator())
        thresh_textline3.setText("129")
        thresh_textline3.setStyleSheet("background: #454545;"
                                       "border: 0px;"
                                       "color: white;"
                                       "border-radius: 5px;"
                                       "font-family: Verdana;")
        thresh_textline3.textEdited.connect(self.text2sld3)

        thresh_segment_button3 = QPushButton()
        thresh_segment_button3.setFixedSize(180, 20)
        thresh_segment_button3.setText("Threshold")
        thresh_segment_button3.setStyleSheet("background: #454545;"
                                             "color: white;"
                                             "border-radius: 5px;"
                                             "font-family: Verdana;")
        segmenter_name = 'binary_thresholding'
        # thresh_segment_button.clicked.connect(lambda: self.segment(segmenter_name, threshold=self.thresh_sld.value()))

        segment_algorithm3_layout1.addWidget(thresh_label3)
        segment_algorithm3_layout1.setAlignment(Qt.AlignLeft)
        segment_algorithm3_layout2.addWidget(thresh_sld3)
        segment_algorithm3_layout2.addWidget(thresh_textline3)
        segment_algorithm3_layout2.setAlignment(Qt.AlignCenter)
        segment_algorithm3_layout3.addWidget(thresh_segment_button3)
        segment_algorithm3_layout3.setAlignment(Qt.AlignRight)

        segment_algorithm3_layout.addLayout(segment_algorithm3_layout1)
        segment_algorithm3_layout.addLayout(segment_algorithm3_layout2)
        segment_algorithm3_layout.addLayout(segment_algorithm3_layout3)
        segment_algorithm3_layout.setAlignment(Qt.AlignTop)
        segment_algorithm3_layout.setSpacing(5)
        segment_algorithm3_layout.setContentsMargins(5, 5, 5, 5)

        self.segment_algorithm3     = segment_algorithm3
        self.thresh_sld3            = thresh_sld3
        self.thresh_label3          = thresh_label3
        self.thresh_textline3       = thresh_textline3
        self.thresh_segment_button3 = thresh_segment_button3

    def init_seg4(self):
        segment_algorithm4 = QWidget()
        segment_algorithm4.setStyleSheet("border: 0px;"
                                         "background: #323232;")

        segment_algorithm4_layout = QVBoxLayout(segment_algorithm4)
        segment_algorithm4_layout1 = QHBoxLayout()
        segment_algorithm4_layout2 = QHBoxLayout()
        segment_algorithm4_layout3 = QHBoxLayout()

        thresh_sld4 = QSlider(Qt.Horizontal)
        thresh_sld4.setMinimum(0)
        thresh_sld4.setMaximum(255)
        thresh_sld4.setValue(129)
        thresh_sld4.setStyleSheet(self.sld_stylesheet)
        thresh_sld4.valueChanged.connect(self.sld2text4)

        thresh_label4 = QLabel()
        thresh_label4.setText("Threshold(0~255)")
        thresh_label4.setAlignment(Qt.AlignCenter)
        thresh_label4.setStyleSheet("font-family: Verdana;"
                                    "color: white;")

        thresh_textline4 = QLineEdit()
        thresh_textline4.setAlignment(Qt.AlignCenter)
        thresh_textline4.setFixedSize(50, 15)
        thresh_textline4.setValidator(QIntValidator())
        thresh_textline4.setText("129")
        thresh_textline4.setStyleSheet("background: #454545;"
                                       "border: 0px;"
                                       "color: white;"
                                       "border-radius: 5px;"
                                       "font-family: Verdana;")
        thresh_textline4.textEdited.connect(self.text2sld4)

        thresh_segment_button4 = QPushButton()
        thresh_segment_button4.setFixedSize(180, 20)
        thresh_segment_button4.setText("Threshold")
        segmenter_name = 'binary_thresholding'
        thresh_segment_button4.setStyleSheet("background: #454545;"
                                             "color: white;"
                                             "border-radius: 5px;"
                                             "font-family: Verdana;")
        # thresh_segment_button.clicked.connect(lambda: self.segment(segmenter_name, threshold=self.thresh_sld.value()))

        segment_algorithm4_layout1.addWidget(thresh_label4)
        segment_algorithm4_layout1.setAlignment(Qt.AlignLeft)
        segment_algorithm4_layout2.addWidget(thresh_sld4)
        segment_algorithm4_layout2.addWidget(thresh_textline4)
        segment_algorithm4_layout2.setAlignment(Qt.AlignCenter)
        segment_algorithm4_layout3.addWidget(thresh_segment_button4)
        segment_algorithm4_layout3.setAlignment(Qt.AlignRight)

        segment_algorithm4_layout.addLayout(segment_algorithm4_layout1)
        segment_algorithm4_layout.addLayout(segment_algorithm4_layout2)
        segment_algorithm4_layout.addLayout(segment_algorithm4_layout3)
        segment_algorithm4_layout.setAlignment(Qt.AlignTop)
        segment_algorithm4_layout.setSpacing(5)
        segment_algorithm4_layout.setContentsMargins(5, 5, 5, 5)

        self.segment_algorithm4     = segment_algorithm4
        self.thresh_sld4            = thresh_sld4
        self.thresh_label4          = thresh_label4
        self.thresh_textline4       = thresh_textline4
        self.thresh_segment_butto4  = thresh_segment_button4

    def init_user_defined(self):
        segment_algorithm5 = QWidget()
        segment_algorithm5.setStyleSheet("border: 0px;"
                                         "background: #323232;")

        segment_algorithm5_layout = QVBoxLayout(segment_algorithm5)
        segment_algorithm5_layout1 = QHBoxLayout()
        segment_algorithm5_layout2 = QHBoxLayout()
        segment_algorithm5_layout3 = QHBoxLayout()

        thresh_sld5 = QSlider(Qt.Horizontal)
        thresh_sld5.setMinimum(0)
        thresh_sld5.setMaximum(255)
        thresh_sld5.setValue(129)
        thresh_sld5.setStyleSheet(self.sld_stylesheet)
        thresh_sld5.valueChanged.connect(self.sld2text5)

        thresh_label5 = QLabel()
        thresh_label5.setText("Threshold(0~255)")
        thresh_label5.setAlignment(Qt.AlignCenter)
        thresh_label5.setStyleSheet("font-family: Verdana;"
                                    "color: white;")

        thresh_textline5 = QLineEdit()
        thresh_textline5.setAlignment(Qt.AlignCenter)
        thresh_textline5.setFixedSize(50, 15)
        thresh_textline5.setValidator(QIntValidator())
        thresh_textline5.setText("129")
        thresh_textline5.setStyleSheet("background: #454545;"
                                       "border: 0px;"
                                       "color: white;"
                                       "border-radius: 5px;"
                                       "font-family: Verdana;")
        thresh_textline5.textEdited.connect(self.text2sld5)

        thresh_segment_button5 = QPushButton()
        thresh_segment_button5.setFixedSize(180, 20)
        thresh_segment_button5.setText("Threshold")
        segmenter_name = 'binary_thresholding'
        thresh_segment_button5.setStyleSheet("background: #454545;"
                                             "color: white;"
                                             "border-radius: 5px;"
                                             "font-family: Verdana;")
        # thresh_segment_button.clicked.connect(lambda: self.segment(segmenter_name, threshold=self.thresh_sld.value()))

        segment_algorithm5_layout1.addWidget(thresh_label5)
        segment_algorithm5_layout1.setAlignment(Qt.AlignLeft)
        segment_algorithm5_layout2.addWidget(thresh_sld5)
        segment_algorithm5_layout2.addWidget(thresh_textline5)
        segment_algorithm5_layout2.setAlignment(Qt.AlignCenter)
        segment_algorithm5_layout3.addWidget(thresh_segment_button5)
        segment_algorithm5_layout3.setAlignment(Qt.AlignRight)

        segment_algorithm5_layout.addLayout(segment_algorithm5_layout1)
        segment_algorithm5_layout.addLayout(segment_algorithm5_layout2)
        segment_algorithm5_layout.addLayout(segment_algorithm5_layout3)
        segment_algorithm5_layout.setAlignment(Qt.AlignTop)
        segment_algorithm5_layout.setSpacing(5)
        segment_algorithm5_layout.setContentsMargins(5, 5, 5, 5)

        self.segment_algorithm5 = segment_algorithm5
        self.thresh_sld5 = thresh_sld5
        self.thresh_label5 = thresh_label5
        self.thresh_textline5 = thresh_textline5
        self.thresh_segment_butto5 = thresh_segment_button5


    def sld2text1(self):
        f = self.thresh_sld1.value()
        self.thresh_textline1.setText(str(f))

    def text2sld1(self):
        t = self.thresh_textline1.text()
        self.thresh_sld1.setValue(int(t))

    def sld2text2(self):
        f = self.thresh_sld2.value()/10
        self.thresh_textline2.setText(str(f))

    def text2sld2(self):
        t = self.thresh_textline2.text()
        self.thresh_sld2.setValue(int(t)*10)

    def sld2text3(self):
        f = self.thresh_sld3.value()
        self.thresh_textline3.setText(str(f))

    def text2sld3(self):
        t = self.thresh_textline3.text()
        self.thresh_sld3.setValue(int(t))

    def sld2text4(self):
        f = self.thresh_sld4.value()
        self.thresh_textline4.setText(str(f))

    def text2sld4(self):
        t = self.thresh_textline4.text()
        self.thresh_sld4.setValue(int(t))

    def sld2text5(self):
        f = self.thresh_sld5.value()
        self.thresh_textline5.setText(str(f))

    def text2sld5(self):
        t = self.thresh_textline5.text()
        self.thresh_sld5.setValue(int(t))
