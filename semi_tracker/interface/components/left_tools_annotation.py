# -*- coding: UTF-8 -*-

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from ..utils import get_icon


class AnnotationTools(QWidget):
    def __init__(self):
        super().__init__()

        self.annotation_tools_stylesheet = """
        QLabel
            {
                font-family: Verdana;
                font-size: 12px;
                color: rgb(245, 245, 245, 255);
            }
        """

        self.annotation_tools = QWidget()
        self.annotation_tools.setStyleSheet(self.annotation_tools_stylesheet)

        self.setup_ui()

    def setup_ui(self):
        origin_path_label = QLabel()
        origin_path_label.setText("Origin path: ")

        origin_path_show_lineedit = QLineEdit()
        origin_path_show_lineedit.setPlaceholderText("Origin path")
        # origin_path_show_lineedit.setFixedSize(180, 20)
        origin_path_show_lineedit.setEnabled(False)
        origin_path_show_lineedit.setStyleSheet("background: #454545;"
                                                "border: 0px;"
                                                "color: white;"
                                                "border-radius: 5px;"
                                                "font-family: Verdana;"
                                                "font-size: 10px;")

        result_path_label = QLabel()
        result_path_label.setText("Result path: ")

        result_path_show_lineedit = QLineEdit()
        result_path_show_lineedit.setPlaceholderText("Result path")
        # result_path_show_lineedit.setFixedSize(180, 20)
        result_path_show_lineedit.setEnabled(False)
        result_path_show_lineedit.setStyleSheet("background: #454545;"
                                                "border: 0px;"
                                                "color: white;"
                                                "border-radius: 5px;"
                                                "font-family: Verdana;"
                                                "font-size: 10px;")

        origin_path_browse_button = QPushButton()
        origin_path_browse_button.setIcon(QIcon((get_icon("browse.png"))))
        origin_path_browse_button.setIconSize(QSize(15, 15))
        origin_path_browse_button.setFlat(True)
        origin_path_browse_button.setStyleSheet("border: 0px")
        origin_path_browse_button.setFixedSize(20, 20)
        origin_path_browse_button.setStyleSheet("background: #454545;"
                                                "color: white;"
                                                "border-radius: 5px;"
                                                "font-family: Verdana;")

        result_path_browse_button = QPushButton()
        result_path_browse_button.setIcon(QIcon((get_icon("browse.png"))))
        result_path_browse_button.setIconSize(QSize(15, 15))
        result_path_browse_button.setFlat(True)
        result_path_browse_button.setStyleSheet("border: 0px")
        result_path_browse_button.setFixedSize(20, 20)
        result_path_browse_button.setStyleSheet("background: #454545;"
                                                "color: white;"
                                                "border-radius: 5px;"
                                                "font-family: Verdana;")

        save_annotation_button = QPushButton()
        save_annotation_button.setText("Save")
        save_annotation_button.setFixedSize(60, 20)
        save_annotation_button.setStyleSheet("background: #454545;"
                                             "color: white;"
                                             "border-radius: 5px;"
                                             "font-family: Verdana;")

        finish_annotation_button = QPushButton()
        finish_annotation_button.setText("Finish")
        finish_annotation_button.setFixedSize(60, 20)
        finish_annotation_button.setStyleSheet("background: #454545;"
                                               "color: white;"
                                               "border-radius: 5px;"
                                               "font-family: Verdana;")

        last_ins_color_label = QLabel()
        last_ins_color_label.setText("color")
        last_ins_name_label = QLabel()
        last_ins_name_label.setText("name")

        origin_path_layout = QHBoxLayout()
        result_path_layout = QHBoxLayout()
        buttons_layout = QHBoxLayout()
        last_ins_layout = QHBoxLayout()

        main_layout = QVBoxLayout()

        # origin_path_layout.addWidget(origin_path_label)
        origin_path_layout.addWidget(origin_path_show_lineedit)
        origin_path_layout.addWidget(origin_path_browse_button)

        # result_path_layout.addWidget(result_path_label)
        result_path_layout.addWidget(result_path_show_lineedit)
        result_path_layout.addWidget(result_path_browse_button)

        buttons_layout.addWidget(save_annotation_button)
        buttons_layout.addWidget(finish_annotation_button)

        # last_ins_layout.addWidget(last_ins_color_label)
        # last_ins_layout.addWidget(last_ins_name_label)

        main_layout.addLayout(origin_path_layout)
        main_layout.addLayout(result_path_layout)
        main_layout.addLayout(buttons_layout)
        main_layout.addLayout(last_ins_layout)
        main_layout.setAlignment(Qt.AlignTop)

        # self.origin_path_label = origin_path_label
        self.origin_path_show_lineedit = origin_path_show_lineedit
        self.origin_path_browse_button = origin_path_browse_button
        # self.result_path_label = result_path_label
        self.result_path_show_lineedit = result_path_show_lineedit
        self.result_path_browse_button = result_path_browse_button
        self.save_annotation_button = save_annotation_button
        self.finish_annotation_button = finish_annotation_button
        self.last_ins_color_label = last_ins_color_label
        self.last_ins_name_label = last_ins_name_label
        self.annotation_tools.setLayout(main_layout)
