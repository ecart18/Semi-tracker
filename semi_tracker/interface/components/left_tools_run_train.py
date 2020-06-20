from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *


class RunTrain(QWidget):
    def __init__(self):
        super().__init__()

        self.trainer_stylesheet = """
                QLabel
                    {
                        font-family: Verdana;
                        font-size: 12px;
                        color: rgb(245, 245, 245, 255);
                    }
                """

        self.run_train = QWidget()
        self.run_train.setStyleSheet(self.trainer_stylesheet)

        self.setup_ui()

    def setup_ui(self):
        run_train_layout = QVBoxLayout()

        load_model_button = QPushButton()
        load_model_button.setText("Load model")
        load_model_button.setStyleSheet("background: #454545;"
                                  "color: rgb(245, 245, 245, 255);")
        load_model_button.setContentsMargins(0, 0, 0, 0)
        load_model_button.setFixedHeight(25)

        select_model_label = QLabel()
        select_model_label.setText("You can train with previous model.")
        select_model_label.setContentsMargins(0, 0, 0, 0)

        model_path_show_lineedit = QLineEdit()
        model_path_show_lineedit.setPlaceholderText("No select model")
        # result_path_show_lineedit.setFixedSize(180, 20)
        model_path_show_lineedit.setEnabled(False)
        model_path_show_lineedit.setStyleSheet("background: #454545;"
                                                "border: 0px;"
                                                "color: white;"
                                                "border-radius: 5px;"
                                                "font-family: Verdana;"
                                                "font-size: 10px;")

        train_model_button = QPushButton()
        train_model_button.setText("Train model")
        train_model_button.setStyleSheet("background: #1E90FF;"
                                        "color: rgb(245, 245, 245, 255);")
        train_model_button.setContentsMargins(0, 0, 0, 0)
        train_model_button.setFixedHeight(25)

        run_train_layout.addWidget(select_model_label)
        run_train_layout.addWidget(load_model_button)
        run_train_layout.addWidget(model_path_show_lineedit)
        run_train_layout.addWidget(train_model_button)
        run_train_layout.setAlignment(Qt.AlignTop)
        run_train_layout.setSpacing(15)

        # self.origin_path_label = origin_path_label
        self.load_model_button = load_model_button
        self.train_model_button = train_model_button
        self.model_path_show_lineedit = model_path_show_lineedit

        self.run_train.setLayout(run_train_layout)