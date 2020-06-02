# -*- coding: UTF-8 -*-

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


class InstanceSettings(QWidget):
    """
    The InstanceSettings is the popup windows when add cell button is clicked.
    """
    def __init__(self, colors, raw_img, cell_id, frame_id):
        QWidget.__init__(self)

        self.setFixedSize(240, 240)

        self.name = "add_" + str(cell_id)
        self.color = colors[0]
        self.raw_img = raw_img
        self.mask = np.zeros(np.shape(raw_img))
        self.colors = colors
        self.n = 0
        self.sample_id = []
        self.ins = Instance(frame_id, cell_id)
        self.setStyleSheet("background: #545454;")

        for i in range(25):
            self.sample_id.append(random.randint(0, 11))

        self.color = self.color[self.sample_id[0]]

        color_label = QLabel()
        color_label.setFixedSize(QSize(30, 30))
        cl_s = "background: rgb(" + str(self.color[0]) + ", " + str(self.color[1]) + ", " + str(self.color[2]) + ");"
        color_label.setStyleSheet(cl_s)

        name_editer = QLineEdit()
        name_editer.setStyleSheet("background: #323232;"
                                  "border: 0px;"
                                  "color: white")
        name_editer.setFixedSize(QSize(100, 30))
        name_editer.setText(self.name)
        name_editer.textChanged.connect(self.name_editer_fnc)

        confirm_button = QPushButton()
        confirm_button.setIcon(QIcon((get_icon("check.png"))))
        confirm_button.setIconSize(QSize(28, 28))
        confirm_button.setFixedSize(QSize(30, 30))
        confirm_button.setFlat(True)
        confirm_button.setStyleSheet("border: 0px")
        confirm_button.clicked.connect(self.confirm_button_fnc)

        color_widget_stylesheet = """
            QTableWidget
            {
                background: transparent;
                border: 0px solid #121212;
            }

            QTableWidget::Item
            {   
                padding-left: 0px;
                padding-right: 0px;
                border: none;
            }
        """
        color_widget = QTableWidget(5, 5)
        color_widget.setFixedSize(QSize(180, 180))
        color_widget.setStyleSheet(color_widget_stylesheet)
        color_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        color_widget.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        color_widget.resizeColumnsToContents()
        color_widget.resizeRowsToContents()
        color_widget.setIconSize(QSize(34, 34))
        color_widget.setShowGrid(False)
        color_widget.horizontalHeader().setVisible(False)
        color_widget.verticalHeader().setVisible(False)
        color_widget.itemClicked.connect(self.color_widget_fnc)

        for i in range(25):
            x = i / 5
            y = i % 5

            new_item = QTableWidgetItem("")
            pix = QPixmap(34, 34)
            ccl = self.colors[i]

            cl = ccl[self.sample_id[i]]
            pix.fill(QColor(cl[0], cl[1], cl[2]))
            new_item.setIcon(QIcon(pix))
            new_item.setSizeHint(QSize(34, 34))
            color_widget.setItem(x, y, new_item)

        sub_layout = QHBoxLayout()
        sub_layout.addWidget(color_label)
        sub_layout.addWidget(name_editer)
        sub_layout.addWidget(confirm_button)

        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)
        main_layout.addLayout(sub_layout)
        main_layout.addWidget(color_widget)

        self.setLayout(main_layout)
        self.setWindowIcon(QIcon(get_icon("cell.png")))
        self.setWindowTitle("Add cell")

        self.name_editer = name_editer
        self.color_label = color_label
        self.confirm_button = confirm_button
        self.color_widget = color_widget

    def name_editer_fnc(self):
        self.name = self.name_editer.text()

    def color_widget_fnc(self):
        y = self.color_widget.currentColumn()
        x = self.color_widget.currentRow()
        ind = x* 5 + y
        sub_c = self.colors[ind]
        self.color = sub_c[self.sample_id[ind]]
        cl_s = "background: rgb(" + str(self.color[0]) + ", " + str(self.color[1]) + ", " + str(self.color[2]) + ");"
        self.color_label.setStyleSheet(cl_s)

    def confirm_button_fnc(self):
        self.ins.color = self.color
        self.ins.name = self.name
        self.ins.raw_img = self.raw_img
        self.close()


class ChangedInstances(object):
    """
    The ChangeInstances is class for a set of updating data
    """
    def __init__(self):
        self.update_ins_id = []
        self._update_ins_label = []
        self._update_ins_name = []
        self._update_ins_color = []

        self.delete_ins_id = []
        self._delete_ins_label = []

    @property
    def update_ins_label(self):
        return self._update_ins_label

    @property
    def update_ins_name(self):
        return self._update_ins_name

    @property
    def update_ins_color(self):
        return self._update_ins_color

    @property
    def delete_ins_label(self):
        return self._delete_ins_label

    def add_update_ins(self, ins_id, label, name, color):
        if label in self._update_ins_label:
            ind = self._update_ins_label.index(label)
            self._update_ins_name[ind] = name
            self._update_ins_color[ind] = color
        else:
            self.update_ins_id.append(ins_id)
            self._update_ins_label.append(label)
            self._update_ins_name.append(name)
            self._update_ins_color.append(color)

    def add_delete_ins(self, ins_id, label):
        self.delete_ins_id = id
        self._delete_ins_label = label


'''
class ProgressBar(QtGui.QWidget):
    def __init__(self, parent=None, total1=20):
        super(ProgressBar, self).__init__(parent)
        self.name_line1  =  QtGui.QLineEdit()

        self.progressbar  =  QtWidgets.QProgressBar()
        self.progressbar.setFixedSize(400, 10)
        self.progressbar.setMinimum(1)
        self.progressbar.setMaximum(total1)


        main_layout  =  QtGui.QGridLayout()
        main_layout.addWidget(self.progressbar, 0, 0)

        self.setLayout(main_layout)
        self.setWindowTitle("                                        Labeling")
        self.setGeometry(500, 300, 500, 40)


    def update_progressbar(self, val1):
        self.progressbar.setValue(val1)
        qApp.processEvents()
'''


class ProjectWindow(QWidget):
    """
    The ProjectWindow is the popup window when new project button is clicked.
    """
    def __init__(self, default_path):
        QWidget.__init__(self)
        self.path = default_path
        self.main_stylesheet = """
        QWidget
        {
            background: #323232;
        }
        QLabel 
        {
            font-family: Verdana;
            color: rgb(245, 245, 245, 255);
        }
        QLineEdit
        {
            font-family: Verdana;
            border-radius: 4px;
            background: #454545;
            color: rgb(245, 245, 245, 255);
        }
        QPushButton
        {
            font-family: Verdana;
            border-radius: 4px;
            background: #454545;
            color: rgb(245, 245, 245, 255);
        }
        """

        self.setStyleSheet(self.main_stylesheet)
        self.setFixedSize(500, 100)

        location_label = QLabel()
        location_label.setText("Location:")

        location = QLineEdit()
        location.setText(default_path)

        browse_button = QPushButton()
        # browse_button.setText(" Browse ")
        browse_button.setIcon(QIcon((get_icon("browse.png"))))
        browse_button.setIconSize(QSize(15, 15))
        browse_button.setFlat(True)
        browse_button.setFixedSize(20, 20)
        browse_button.setStyleSheet("background: #454545;"
                                    "color: white;"
                                    "border-radius: 5px;"
                                    "font-family: Verdana;"
                                    "border: 0px;")
        browse_button.clicked.connect(self.browse_fnc)

        create1 = QPushButton()
        create1.setText("Create")
        create1.setFixedSize(80, 25)
        create1.clicked.connect(self.create_fnc)

        self.layout1 = QHBoxLayout()
        self.layout1.addWidget(location_label)
        self.layout1.addWidget(location)
        self.layout1.addWidget(browse_button)

        layout2 = QHBoxLayout()
        layout2.addWidget(create1)
        layout2.setAlignment(Qt.AlignRight)

        layout = QVBoxLayout()
        layout.addLayout(self.layout1)
        layout.addLayout(layout2)

        self.setLayout(layout)
        self.setWindowIcon(QIcon(get_icon("cell.png")))
        self.setWindowTitle("Create project")
        self.location_label = location_label
        self.location = location
        self.browse_button = browse_button
        self.create1 = create1

    def browse_fnc(self):
        dir_path = QFileDialog.getExistingDirectory()
        if not dir_path == "":
            seq = [dir_path, "untitled"]
            show_path = "/".join(seq)
            self.location.setText(show_path)

    def create_fnc(self):
        self.path = self.location.text()
        self.close()


class CellAttributeWindow(QWidget):
    """
    The CellAttributeWindow is for displaying more details of cells.
    """
    def __init__(self, ins):
        QWidget.__init__(self)
        # super(CellAttributeWindow).__init__()

        self.frame_id = ins.frame_id
        self.ins = ins
        self.ins_label = ins.label_id
        self.ins_name = ins.name
        self.ins_color = ins.color
        self.ins_bbox = ins.bbox
        self.ins_area = ins.area
        self.ins_centroid = ins.centroid
        self.ins_intensity = ins.intensity

        self.setup_ui()

    def setup_ui(self):

        main_stylesheet = """
        QWidget
        {
            background: #323232;
        }
        QLabel
        {
            font-family: Verdana;
            color: white;
            border-radius: 4px;
        }
        QLineEdit
        {
            font-family: Verdana;
            background: #454545;
            color: white;
            border: 0px;
            border-radius: 4px;
        }
        """

        cell_show = QLabel()
        cell_show.setFixedSize(170, 170)
        self.get_cell_icon()
        cell_show_pix = QPixmap(get_icon("cell_show.png"))
        cell_show_pix = cell_show_pix.scaled(QSize(170, 170), Qt.KeepAspectRatio)
        cell_show.setPixmap(cell_show_pix)

        frame_id = QLabel()
        frame_id_display = "Frame id: " + str(self.frame_id)
        frame_id.setText(frame_id_display)

        cell_label = QLabel()
        cell_label_display = "Cell label: " + str(self.ins_label)
        cell_label.setText(cell_label_display)

        cell_name = QLabel()
        cell_name.setText("Cell name: ")
        cell_name_editor = QLineEdit()
        cell_name_editor.setText(str(self.ins_name))
        # cell_name_editor.setFixedSize(40, 20)

        cell_color = QLabel()
        cell_color.setText("Cell color: ")
        stylesheet = "background: rgb(" + str(self.ins_color[0]) + ", " + str(self.ins_color[1]) + ", " \
                     + str(self.ins_color[2]) + ", 255);"
        cell_color_show = QLabel()
        cell_color_show.setStyleSheet(stylesheet)
        cell_color_show.setFixedSize(40, 20)

        cell_bbox = QLabel()
        cell_bbox_display = "Cell bbox: " + str(self.ins_bbox)
        cell_bbox.setText(cell_bbox_display)

        cell_centroid = QLabel()
        cell_centroid_display = "Cell centroid: " + str(self.ins_centroid)
        cell_centroid.setText(cell_centroid_display)

        cell_area = QLabel()
        cell_area_display = "Cell area: " + str(self.ins_area)
        cell_area.setText(cell_area_display)

        cell_intensity = QLabel()
        cell_intensity_display = "Cell intensity: " + str(self.ins_intensity)
        cell_intensity.setText(cell_intensity_display)

        labels_layout = QVBoxLayout()
        labels_layout.addWidget(frame_id)
        labels_layout.addWidget(cell_label)
        name_layout = QHBoxLayout()
        name_layout.addWidget(cell_name)
        name_layout.addWidget(cell_name_editor)
        name_layout.setAlignment(Qt.AlignLeft)
        labels_layout.addLayout(name_layout)
        color_layout = QHBoxLayout()
        color_layout.addWidget(cell_color)
        color_layout.addWidget(cell_color_show)
        color_layout.setAlignment(Qt.AlignLeft)
        labels_layout.addLayout(color_layout)
        labels_layout.addWidget(cell_bbox)
        labels_layout.addWidget(cell_centroid)
        labels_layout.addWidget(cell_area)
        labels_layout.addWidget(cell_intensity)

        main_layout = QHBoxLayout()
        main_layout.addWidget(cell_show)
        main_layout.addLayout(labels_layout)

        self.setFixedSize(450, 220)
        self.setWindowIcon(QIcon(get_icon("cell.png")))
        self.setWindowTitle("Cell attribute")
        self.setStyleSheet(main_stylesheet)
        self.setLayout(main_layout)

    def get_cell_icon(self):
        coords = self.ins.coords
        length = np.shape(coords)[1]
        x = coords[1]-[self.ins_bbox[0]]*length
        y = coords[0]-[self.ins_bbox[1]]*length

        sub_img = np.zeros((self.ins_bbox[3]-self.ins_bbox[1]+1, self.ins_bbox[2]-self.ins_bbox[0]+1, 3))
        sub_img[y, x, 0] = self.ins_color[2]
        sub_img[y, x, 1] = self.ins_color[1]
        sub_img[y, x, 2] = self.ins_color[0]

        cv2.imwrite('./semi_tracker/interface/icon/cell_show.png', sub_img)
