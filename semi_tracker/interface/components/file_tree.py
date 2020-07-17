# -*- coding: UTF-8 -*-

import os
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtCore import QFileInfo, QUrl
from PyQt5 import QtGui
from PyQt5 import QtCore
from ..utils import get_icon, left_tools_stylesheet, instance_widget_stylesheet


class FileTree(QWidget):
    def __init__(self, dir_path=None, view_flag=0):
        super(FileTree, self).__init__()
        self.dir_path = dir_path
        self.view_flag = view_flag
        
        self.setup_ui()

    def setup_ui(self):
        if self.dir_path is None:
            create_button = QPushButton()
            create_button.setText("Open folder")
            create_button.setStyleSheet("background: #454545;"
                                        "color: rgb(245, 245, 245);"
                                        "font-size: 12px;"
                                        "font-family: Verdana;")
            create_button.setContentsMargins(0, 0, 0, 0)
            create_button.setFixedHeight(25)

            label = QLabel()
            label.setText("You have not yet set project folder.")
            label.setStyleSheet("color: rgb(245, 245, 245);"
                                "font-size: 12px;"
                                "font-family: Verdana;")
            label.setContentsMargins(0, 0, 0, 0)

            label1 = QLabel()
            label1.setText("Otherwise the default folder is:")
            label1.setStyleSheet("color: rgb(245, 245, 245);"
                                "font-size: 12px;"
                                "font-family: Verdana;")
            label1.setContentsMargins(0, 0, 0, 0)

            default_button = QPushButton()
            default_button.setText("Use .output/untitled")
            default_button.setStyleSheet("background: #454545;"
                                        "color: rgb(245, 245, 245);"
                                        "font-size: 12px;"
                                        "font-family: Verdana;")
            default_button.setContentsMargins(0, 0, 0, 0)
            default_button.setFixedHeight(25)

            load_button = QPushButton()
            load_button.setText("Load images")
            load_button.setStyleSheet("background: #454545;"
                                        "color: rgb(245, 245, 245);"
                                        "font-size: 12px;"
                                        "font-family: Verdana;")
            load_button.setContentsMargins(0, 0, 0, 0)
            load_button.setFixedHeight(25)

            label2 = QLabel()
            label2.setText("Load images or set image files folder.")
            label2.setStyleSheet("color: rgb(245, 245, 245);"
                                "font-size: 12px;"
                                "font-family: Verdana;")
            label2.setContentsMargins(0, 0, 0, 0)

            layout1 = QVBoxLayout()
            layout1.addWidget(label2)
            layout1.addWidget(load_button)
            layout1.addWidget(label)
            layout1.addWidget(create_button)
            layout1.addWidget(label1)
            layout1.addWidget(default_button)
            layout1.setSpacing(10)

            layout2 = QVBoxLayout()
            layout2.addLayout(layout1)
            layout2.setAlignment(QtCore.Qt.AlignTop)

            self.setLayout(layout2)
            self.create_button = create_button
            self.default_button = default_button
            self.load_button = load_button

        elif self.view_flag == 1 and self.dir_path is not None:
            self.tool_box = QToolBox()

            self.widget1 = QWidget()
            create_button = QPushButton()
            create_button.setText("Open folder")
            create_button.setStyleSheet("background: #454545;"
                                        "color: rgb(245, 245, 245);"
                                        "font-size: 12px;"
                                        "font-family: Verdana;"
                                        "border: 0px;")
            create_button.setContentsMargins(0, 0, 0, 0)
            create_button.setFixedHeight(25)

            label = QLabel()
            label.setText("You have not yet set project folder.")
            label.setStyleSheet("color: rgb(245, 245, 245);"
                                "font-size: 12px;"
                                "font-family: Verdana;")
            label.setContentsMargins(0, 0, 0, 0)

            label1 = QLabel()
            label1.setText("Otherwise the default folder is:")
            label1.setStyleSheet("color: rgb(245, 245, 245);"
                                 "font-size: 12px;"
                                 "font-family: Verdana;")
            label1.setContentsMargins(0, 0, 0, 0)

            default_button = QPushButton()
            default_button.setText("Use .output/untitled")
            default_button.setStyleSheet("background: #454545;"
                                         "color: rgb(245, 245, 245);"
                                         "font-size: 12px;"
                                         "font-family: Verdana;"
                                         "border: 0px;")
            default_button.setContentsMargins(0, 0, 0, 0)
            default_button.setFixedHeight(25)

            load_button = QPushButton()
            load_button.setText("Load images")
            load_button.setStyleSheet("background: #454545;"
                                      "color: rgb(245, 245, 245);"
                                      "font-size: 12px;"
                                      "font-family: Verdana;"
                                      "border: 0px;")
            load_button.setContentsMargins(0, 0, 0, 0)
            load_button.setFixedHeight(25)

            label2 = QLabel()
            label2.setText("Load images or set image files folder.")
            label2.setStyleSheet("color: rgb(245, 245, 245);"
                                 "font-size: 12px;"
                                 "font-family: Verdana;")
            label2.setContentsMargins(0, 0, 0, 0)

            layout1 = QVBoxLayout()
            layout1.addWidget(label2)
            layout1.addWidget(load_button)
            layout1.addWidget(label)
            layout1.addWidget(create_button)
            layout1.addWidget(label1)
            layout1.addWidget(default_button)
            layout1.setSpacing(10)

            layout2 = QVBoxLayout()
            layout2.addLayout(layout1)
            layout2.setAlignment(QtCore.Qt.AlignTop)

            self.widget1.setLayout(layout2)
            self.widget1.setStyleSheet("background: #323232;"
                                       "padding: 0px;")
            self.create_button = create_button
            self.default_button = default_button
            self.load_button = load_button

            self.widget2 = QWidget()
            self.widget2.setStyleSheet("background: #323232;")
            tree_stylesheet = """
                        QTreeWidget 
                        {
                            font-family: Verdana;
                            color: rgb(245, 245, 245);
                            border: 0px;
                        }
                        """
            tree = QTreeWidget()
            tree.doubleClicked.connect(self.open_file)
            tree.setStyleSheet(tree_stylesheet)
            tree.setColumnCount(1)
            tree.setColumnWidth(0, 50)
            tree.setHeaderHidden(True)
            tree.setHeaderLabel("EXPLORE")
            tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
            tree.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

            dirs = os.listdir(self.dir_path)

            file_info = QFileInfo(self.dir_path)
            file_icon = QFileIconProvider()
            icon = QtGui.QIcon(file_icon.icon(file_info))
            tree_root = QTreeWidgetItem(tree)
            tree_root.setText(0, self.dir_path.split('/')[-1])
            tree_root.setIcon(0, QtGui.QIcon(icon))
            self.create_tree(dirs, tree_root, self.dir_path)
            layout2 = QHBoxLayout()
            layout2.addWidget(tree)
            self.widget2.setLayout(layout2)

            self.tree = tree
            self.tool_box.addItem(self.widget1, "Basic settings")
            self.tool_box.addItem(self.widget2, "File system")
            self.tool_box.setStyleSheet(left_tools_stylesheet)
            main_layout = QHBoxLayout()
            main_layout.addWidget(self.tool_box)
            main_layout.setContentsMargins(0, 0, 0, 0)
            self.setLayout(main_layout)

        else:
            tree_stylesheet = """
            QTreeWidget 
            {
                font-family: Verdana;
                color: rgb(245, 245, 245);
            }
            """
            tree = QTreeWidget()
            tree.doubleClicked.connect(self.open_file)
            tree.setStyleSheet(tree_stylesheet)
            tree.setColumnCount(1)
            tree.setColumnWidth(0, 50)
            tree.setHeaderHidden(True)
            tree.setHeaderLabel("EXPLORE")
            tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
            tree.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

            dirs = os.listdir(self.dir_path)

            file_info = QFileInfo(self.dir_path)
            file_icon = QFileIconProvider()
            icon = QtGui.QIcon(file_icon.icon(file_info))
            tree_root = QTreeWidgetItem(tree)
            tree_root.setText(0, self.dir_path.split('/')[-1])
            tree_root.setIcon(0, QtGui.QIcon(icon))
            self.create_tree(dirs, tree_root, self.dir_path)
            layout2 = QHBoxLayout()
            layout2.addWidget(tree)
            self.setLayout(layout2)

            self.tree = tree

    def create_tree(self, dirs, root, path):
        for dir in dirs:
            new_path = os.path.join(path, dir)
            file_info = QFileInfo(new_path)
            file_icon = QFileIconProvider()
            icon = QtGui.QIcon(file_icon.icon(file_info))
            child_node = QTreeWidgetItem(root)
            child_node.setText(0, dir)
            child_node.setIcon(0, QtGui.QIcon(icon))
            if os.path.isdir(new_path):
                new_dirs = os.listdir(new_path)
                self.create_tree(new_dirs, child_node, new_path)

    def open_file(self):
        url_list = [self.tree.currentItem().text(0)]
        parent_item = self.tree.currentItem().parent()
        while parent_item is not None:
            url_list.append(parent_item.text(0))
            parent_item = parent_item.parent()
        url_list = url_list[:-1]
        url = "/".join(url_list[::-1])
        url = "/".join([self.dir_path, url])
        url_info = QFileInfo(url)
        if not url_info.isDir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(url))
