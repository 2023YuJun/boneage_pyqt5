import sys
import os
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os
import time
import cv2

from PyQt5.QtCore import Qt, QPoint, QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu, QAction
from PyQt5.QtGui import QImage, QPixmap, QPainter, QIcon
from PyQt5 import uic
from mainUI.mainUI import Ui_MainWindow


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.m_flag = False

        # style 1: 窗体可以拉伸
        # self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowStaysOnTopHint)

        # style 2: 窗体不可以拉伸
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint
                            | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)

        # self.setWindowOpacity(0.85)  # 窗体透明度
        self.maxButton.setCheckable(True)  # 可选择
        self.maxButton.clicked.connect(self.max_or_restore)
        # 直接显示最大化窗体
        # self.maxButton.animateClick(10)

        self.fileButton.clicked.connect(self.open_file)

        self.cameraButton.setCheckable(True)
        self.cameraButton.clicked.connect(self.toggle_camera)
        # 初始化摄像头和计时器
        self.capture = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # 更改读条
        self.confSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'confSpinBox'))
        self.confSlider.valueChanged.connect(lambda x: self.change_val(x, 'confSlider'))
        self.iouSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'iouSpinBox'))
        self.iouSlider.valueChanged.connect(lambda x: self.change_val(x, 'iouSlider'))
        self.rateSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'rateSpinBox'))
        self.rateSlider.valueChanged.connect(lambda x: self.change_val(x, 'rateSlider'))

    def max_or_restore(self):
        if self.maxButton.isChecked():
            self.window().showMaximized()
        else:
            self.window().showNormal()

    def open_file(self):

        config_file = 'config/Default_path.json'
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['Default_path']
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, 'Video/image', open_fold, "Pic_File(*.mp4 *.mkv *.avi *.flv "
                                                                              "*.jpg *.png)")
        if name:
            self.det_thread.source = name
            self.statistic_msg('Loaded file：{}'.format(os.path.basename(name)))
            config['Default_path'] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            self.stop()

    def toggle_camera(self):
        if self.cameraButton.isChecked():
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        if self.capture is None:
            self.capture = cv2.VideoCapture(0)  # 打开默认摄像头（索引为0）
        self.timer.start(20)  # 设置更新画面的时间间隔

    def stop_camera(self):
        self.timer.stop()
        if self.capture is not None:
            self.capture.release()  # 释放摄像头
            self.capture = None  # 将 self.capture 设为 None
        self.label_previous.clear()

    def update_frame(self):
        ret, frame = self.capture.read()  # 读取摄像头画面

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 将图像格式从 OpenCV 格式转换为 RGB 格式
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(convert_to_Qt_format)
            self.label_previous.setPixmap(pixmap.scaled(self.label_previous.size()))  # 在 QLabel 中显示画面

    def closeEvent(self, event):
        if self.capture is not None:
            self.capture.release()  # 释放摄像头
        event.accept()

    def change_val(self, x, flag):
        if flag == 'confSpinBox':
            self.confSlider.setValue(int(x * 100))
        elif flag == 'confSlider':
            self.confSpinBox.setValue(x / 100)
        elif flag == 'iouSpinBox':
            self.iouSlider.setValue(int(x * 100))
        elif flag == 'iouSlider':
            self.iouSpinBox.setValue(x / 100)
        elif flag == 'rateSpinBox':
            self.rateSlider.setValue(x)
        elif flag == 'rateSlider':
            self.rateSpinBox.setValue(x)
        else:
            pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myForm = MainWindow()
    myForm.show()
    app.exec()