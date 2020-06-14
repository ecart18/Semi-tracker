# -*- coding: UTF-8 -*-

import os
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtCore import QFileInfo, QUrl
# from PyQt5 import *
from PyQt5 import QtGui
from PyQt5 import QtCore


class FileTree(QWidget):
    def __init__(self, dir_path=None):
        super(FileTree, self).__init__()
        # need an absolute path
        self.open_path = "D:/Semi-tracker/semi-tracker/output"
        self.dir_path = dir_path
        
        self.setup_ui()
        # self.show()

    def setup_ui(self):
        if self.dir_path is None:
            create_button = QPushButton()
            create_button.setText("Create")
            create_button.setStyleSheet("color: rgb(30, 144, 255, 255);"
                                        "font-size: 12px;"
                                        "font-family: Verdana;")
            create_button.setContentsMargins(0, 0, 0, 0)

            label = QLabel()
            label.setText("a new project.")
            # label.setAlignment(QtCore.Qt.AlignRight)
            label.setStyleSheet("color: rgb(245, 245, 245, 255);"
                                "font-size: 12px;"
                                "font-family: Verdana;")
            label.setContentsMargins(0, 0, 0, 0)

            layout1 = QHBoxLayout()
            layout1.addWidget(create_button)
            layout1.addWidget(label)
            layout1.setSpacing(0)
            # layout1.addWidget(or_label)
            # layout1.addWidget(open_button)

            layout2 = QVBoxLayout()
            layout2.addLayout(layout1)
            # layout2.addWidget(label)
            layout2.setAlignment(QtCore.Qt.AlignTop)

            self.setLayout(layout2)
            self.create_button = create_button

        else:
            # 格式检查
            # https://blog.csdn.net/qq_41398808/article/details/102838427
            tree_stylesheet = """
            QTreeWidget 
            {
                font-family: Verdana;
                color: rgb(245, 245, 245, 255);
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
        url = "/".join(url_list[::-1])
        # print(url)

        full_url = "file:///" + self.open_path + "/" + url
        # print(full_url)
        url_info = QFileInfo(self.open_path + "/" + url)
        if not url_info.isDir():
            QDesktopServices.openUrl(QUrl(full_url))
