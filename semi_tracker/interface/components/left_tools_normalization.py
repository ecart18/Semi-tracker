import os.path as osp
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from ..utils import get_icon


class NormalizeTools(QWidget):
    def __init__(self):
        super().__init__()

        self.normalize_tools_stylesheet = """
            QToolBox 
            {
                background: #282828;
                padding-bottom: 0px;
            }
            QToolBox::tab 
            {
                font-family: Verdana;
                font-size: 12px;
                background: #666666;
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
        self.normalize_tools = QToolBox()
        # self.normalize_tools.clearFocus()
        self.normalize_tools.setStyleSheet(self.normalize_tools_stylesheet)
        self.normalize_tools.layout().setSpacing(1)

        self.setup_ui()

    def setup_ui(self):
        self.init_equalize_hist()
        self.init_min_max()
        self.init_retinex_MSRCP()
        self.init_retinex_MSRCR()

        self.normalize_tools.addItem(self.equalize_hist, "Equalize Histogram")
        self.normalize_tools.addItem(self.min_max, "Min-Max")
        self.normalize_tools.addItem(self.retinex_MSRCP, "Retinex-MSRCP")
        self.normalize_tools.addItem(self.retinex_MSRCR, "Retinex-MSRCR")
        self.normalize_tools.setItemIcon(0, QIcon(get_icon("Arrow_down.png")))
        self.normalize_tools.setItemIcon(1, QIcon(get_icon("Arrow_right.png")))
        self.normalize_tools.setItemIcon(2, QIcon(get_icon("Arrow_right.png")))
        self.normalize_tools.setItemIcon(3, QIcon(get_icon("Arrow_right.png")))

    def init_equalize_hist(self):
        equalize_hist = QWidget()
        equalize_hist.setStyleSheet("border: 0px;"
                                    "background: #323232;")
        equalize_hist_layout = QHBoxLayout()
        equalize_hist_layout_main = QVBoxLayout()

        equalize_hist_label = QLabel()
        equalize_hist_label.setText("Equalize histogram: ")
        equalize_hist_label.setAlignment(Qt.AlignLeft)
        equalize_hist_label.setStyleSheet("font-family: Verdana;"
                                          "color: white;")

        equalize_hist_button = QPushButton()
        equalize_hist_button.setFixedSize(60, 20)
        equalize_hist_button.setText("Run")
        equalize_hist_button.setStyleSheet("background: #454545;"
                                           "color: white;"
                                           "border-radius: 5px;"
                                           "font-family: Verdana;")

        equalize_hist_layout.addWidget(equalize_hist_label)
        equalize_hist_layout.addWidget(equalize_hist_button)
        equalize_hist_layout_main.addLayout(equalize_hist_layout)
        equalize_hist_layout_main.setAlignment(Qt.AlignTop)
        equalize_hist.setLayout(equalize_hist_layout_main)

        self.equalize_hist = equalize_hist
        self.equalize_hist_label = equalize_hist_label
        self.equalize_hist_button = equalize_hist_button

    def init_min_max(self):
        min_max = QWidget()
        min_max.setStyleSheet("border: 0px;"
                              "background: #323232;")
        min_max_layout_main = QVBoxLayout()
        min_max_layout = QHBoxLayout()

        min_max_label = QLabel()
        min_max_label.setText("Min-Max: ")
        min_max_label.setAlignment(Qt.AlignLeft)
        min_max_label.setStyleSheet("font-family: Verdana;"
                                          "color: white;")

        min_max_button = QPushButton()
        min_max_button.setFixedSize(60, 20)
        min_max_button.setText("Run")
        min_max_button.setStyleSheet("background: #454545;"
                                           "color: white;"
                                           "border-radius: 5px;"
                                           "font-family: Verdana;")

        min_max_layout.addWidget(min_max_label)
        min_max_layout.addWidget(min_max_button)
        min_max_layout_main.addLayout(min_max_layout)
        min_max_layout_main.setAlignment(Qt.AlignTop)
        min_max.setLayout(min_max_layout_main)

        self.min_max = min_max
        self.min_max_label = min_max_label
        self.min_max_button = min_max_button

    def init_retinex_MSRCP(self):
        retinex_MSRCP = QWidget()
        retinex_MSRCP.setStyleSheet("border: 0px;"
                              "background: #323232;")
        retinex_MSRCP_layout_main = QVBoxLayout()
        retinex_MSRCP_layout = QHBoxLayout()

        retinex_MSRCP_label = QLabel()
        retinex_MSRCP_label.setText("Retinex-MSRCP: ")
        retinex_MSRCP_label.setAlignment(Qt.AlignLeft)
        retinex_MSRCP_label.setStyleSheet("font-family: Verdana;"
                                    "color: white;")

        retinex_MSRCP_button = QPushButton()
        retinex_MSRCP_button.setFixedSize(60, 20)
        retinex_MSRCP_button.setText("Run")
        retinex_MSRCP_button.setStyleSheet("background: #454545;"
                                     "color: white;"
                                     "border-radius: 5px;"
                                     "font-family: Verdana;")

        retinex_MSRCP_layout.addWidget(retinex_MSRCP_label)
        retinex_MSRCP_layout.addWidget(retinex_MSRCP_button)
        retinex_MSRCP_layout_main.addLayout(retinex_MSRCP_layout)
        retinex_MSRCP_layout_main.setAlignment(Qt.AlignTop)
        retinex_MSRCP.setLayout(retinex_MSRCP_layout_main)

        self.retinex_MSRCP = retinex_MSRCP
        self.retinex_MSRCP_label = retinex_MSRCP_label
        self.retinex_MSRCP_button = retinex_MSRCP_button

    def init_retinex_MSRCR(self):
        retinex_MSRCR = QWidget()
        retinex_MSRCR.setStyleSheet("border: 0px;"
                              "background: #323232;")
        retinex_MSRCR_layout_main = QVBoxLayout()
        retinex_MSRCR_layout = QHBoxLayout()

        retinex_MSRCR_label = QLabel()
        retinex_MSRCR_label.setText("Retinex-MSRCR: ")
        retinex_MSRCR_label.setAlignment(Qt.AlignLeft)
        retinex_MSRCR_label.setStyleSheet("font-family: Verdana;"
                                    "color: white;")

        retinex_MSRCR_button = QPushButton()
        retinex_MSRCR_button.setFixedSize(60, 20)
        retinex_MSRCR_button.setText("Run")
        retinex_MSRCR_button.setStyleSheet("background: #454545;"
                                     "color: white;"
                                     "border-radius: 5px;"
                                     "font-family: Verdana;")

        retinex_MSRCR_layout.addWidget(retinex_MSRCR_label)
        retinex_MSRCR_layout.addWidget(retinex_MSRCR_button)
        retinex_MSRCR_layout_main.addLayout(retinex_MSRCR_layout)
        retinex_MSRCR_layout_main.setAlignment(Qt.AlignTop)
        retinex_MSRCR.setLayout(retinex_MSRCR_layout_main)

        self.retinex_MSRCR = retinex_MSRCR
        self.retinex_MSRCR_label = retinex_MSRCR_label
        self.retinex_MSRCR_button = retinex_MSRCR_button