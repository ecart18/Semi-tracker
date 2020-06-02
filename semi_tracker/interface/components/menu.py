# -*- coding: UTF-8 -*-

from PyQt5.QtWidgets import *


class Menu(QWidget):

    def __init__(self, parent=None):
        super().__init__()

        self.menu_bar_style_sheet = """
            background-color:#525252;
        """
        self.menu_bar = QMenuBar()
        self.menu_bar.setStyleSheet(self.menu_bar_style_sheet)

        self.setup_ui()

    def setup_ui(self):
        # Initialize menu bar
        self.init_file()
        self.init_edit()
        self.init_run()
        self.init_tools()
        self.init_help()

    def init_file(self):
        # File
        file_menu = self.menu_bar.addMenu("&File")

        new_project_act = QAction("New project...", self)
        new_project_act.setShortcut('Ctrl+N')
        # new_project_act.triggered.connect(self.new_project)
        file_menu.addAction(new_project_act)

        load_act = QAction("&Load...", self)
        load_act.setShortcut('Ctrl+O')
        # load_act.triggered.connect(self.load_files)
        file_menu.addAction(load_act)

        file_menu.addMenu("&Save As...")
        open_recent = file_menu.addMenu("&Open Recent")
        open_recent.addMenu("cell")
        file_menu.addMenu("&Settings...")

        exit_act = QAction("&Exit", self)
        exit_act.setShortcut('Ctrl+Q')
        # exit_act.triggered.connect(self.exit_event)
        file_menu.addAction(exit_act)

        self.file_menu          = file_menu
        self.new_project_act    = new_project_act
        self.load_act           = load_act
        self.open_recent        = open_recent
        self.exit_act           = exit_act

    def init_edit(self):
        edit_menu = self.menu_bar.addMenu("&Edit")
        self.edit_menu = edit_menu

    def init_run(self):
        run_menu = self.menu_bar.addMenu("&Run")
        self.run_menu = run_menu

    def init_tools(self):
        tools_menu = self.menu_bar.addMenu("&Tools")
        self.tools_menu = tools_menu

    def init_help(self):
        help_menu = self.menu_bar.addMenu("&Help")
        self.help_menu = help_menu

