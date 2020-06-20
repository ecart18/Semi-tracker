from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from ..utils import slide_stylesheet


class DataLoader(QWidget):
    def __init__(self):
        super().__init__()

        self.data_loader = QWidget()

        self.setup_ui()

    def setup_ui(self):
        batch_size_label = QLabel()
        batch_size_label.setText("Batch size:")
        batch_size_label.setStyleSheet("font-family: Verdana;"
                                        "color: white;")

        batch_size_select = QComboBox()
        batch_size_select.setFixedSize(80, 20)
        batch_size_select.setStyleSheet("border: 0px;"
                                        "color: white;"
                                        "font-family: Verdana;"
                                        "background: transparent;")
        batch_size_select.addItem("1")
        batch_size_select.addItem("2")
        batch_size_select.addItem("4")
        batch_size_select.addItem("8")
        batch_size_select.addItem("16")
        batch_size_select.addItem("32")
        batch_size_select.addItem("64")
        batch_size_select.setCurrentIndex(2)

        batch_size_layout = QHBoxLayout()
        batch_size_layout.addWidget(batch_size_label)
        batch_size_layout.addWidget(batch_size_select)

        validation_radio_sld = QSlider(Qt.Horizontal)
        validation_radio_sld.setMinimum(1)
        validation_radio_sld.setMaximum(5)
        validation_radio_sld.setValue(2)
        validation_radio_sld.setStyleSheet(slide_stylesheet)
        validation_radio_sld.valueChanged.connect(self.sld2text)

        validation_radio_label = QLabel()
        validation_radio_label.setText("Validation radio(0.1~0.5)")
        validation_radio_label.setAlignment(Qt.AlignLeft)
        validation_radio_label.setStyleSheet("font-family: Verdana;"
                                    "color: white;")

        validation_radio_textline = QLineEdit()
        validation_radio_textline.setAlignment(Qt.AlignCenter)
        validation_radio_textline.setFixedSize(50, 15)
        validation_radio_textline.setValidator(QIntValidator())
        validation_radio_textline.setText("0.2")
        validation_radio_textline.setStyleSheet("background: #454545;"
                                       "border: 0px;"
                                       "color: white;"
                                       "border-radius: 5px;"
                                       "font-family: Verdana;")
        validation_radio_textline.textEdited.connect(self.text2sld)

        validation_radio_layout = QVBoxLayout()
        validation_radio_layout1 = QHBoxLayout()
        validation_radio_layout1.addWidget(validation_radio_sld)
        validation_radio_layout1.addWidget(validation_radio_textline)
        validation_radio_layout.addWidget(validation_radio_label)
        validation_radio_layout.addLayout(validation_radio_layout1)

        paraller_works_label = QLabel()
        paraller_works_label.setText("Paraller works:")
        paraller_works_label.setStyleSheet("font-family: Verdana;"
                                       "color: white;")

        paraller_works_select = QComboBox()
        paraller_works_select.setFixedSize(80, 20)
        paraller_works_select.setStyleSheet("border: 0px;"
                                        "color: white;"
                                        "font-family: Verdana;"
                                        "background: transparent;")
        paraller_works_select.addItem("2")
        paraller_works_select.addItem("4")
        paraller_works_select.addItem("8")
        paraller_works_select.setCurrentIndex(0)

        paraller_works_layout = QHBoxLayout()
        paraller_works_layout.addWidget(paraller_works_label)
        paraller_works_layout.addWidget(paraller_works_select)

        augmentation_label = QLabel()
        augmentation_label.setText("Augmentation:")
        augmentation_label.setStyleSheet("font-family: Verdana;"
                                           "color: white;")

        flip_checkbox = QCheckBox()
        flip_checkbox.setText("Flip")
        flip_checkbox.setStyleSheet("font-family: Verdana;"
                                                   "font-size: 12px;"
                                                   "color: white;")

        rotate_checkbox = QCheckBox()
        rotate_checkbox.setText("Rotate")
        rotate_checkbox.setStyleSheet("font-family: Verdana;"
                                    "font-size: 12px;"
                                    "color: white;")

        gaussian_noise_checkbox = QCheckBox()
        gaussian_noise_checkbox.setText("Gaussian noise")
        gaussian_noise_checkbox.setStyleSheet("font-family: Verdana;"
                                    "font-size: 12px;"
                                    "color: white;")

        gaussian_blur_checkbox = QCheckBox()
        gaussian_blur_checkbox.setText("Gaussian blur")
        gaussian_blur_checkbox.setStyleSheet("font-family: Verdana;"
                                              "font-size: 12px;"
                                              "color: white;")

        augmentation_layout = QVBoxLayout()
        augmentation_layout1 = QVBoxLayout()
        augmentation_layout1.addWidget(flip_checkbox)
        augmentation_layout1.addWidget(rotate_checkbox)
        augmentation_layout1.addWidget(gaussian_noise_checkbox)
        augmentation_layout1.addWidget(gaussian_blur_checkbox)
        augmentation_layout1.setContentsMargins(20, 0, 0, 0)
        augmentation_layout.addWidget(augmentation_label)
        augmentation_layout.addLayout(augmentation_layout1)

        data_loader_layout = QVBoxLayout()
        data_loader_layout.addLayout(batch_size_layout)
        data_loader_layout.addLayout(validation_radio_layout)
        data_loader_layout.addLayout(paraller_works_layout)
        data_loader_layout.addLayout(augmentation_layout)

        self.data_loader.setLayout(data_loader_layout)

        self.batch_size_select = batch_size_select
        self.validation_radio_sld = validation_radio_sld
        self.validation_radio_textline = validation_radio_textline
        self.paraller_works_select = paraller_works_select
        self.flip_checkbox = flip_checkbox
        self.rotate_checkbox = rotate_checkbox
        self.gaussian_noise_checkbox = gaussian_noise_checkbox
        self.gaussian_blur_checkbox = gaussian_blur_checkbox

    def sld2text(self):
        f = self.validation_radio_sld.value() / 10
        self.validation_radio_textline.setText(str(f))

    def text2sld(self):
        t = self.validation_radio_textline.text()
        self.validation_radio_sld.setValue(int(t) * 10)

