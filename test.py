import sys
import os
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os
import time
import cv2
from ultralytics import YOLO
from PIL import Image
from PyQt5.QtCore import Qt, QPoint, QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu, QAction, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap, QPainter, QIcon
from PyQt5 import uic
from mainUI.mainUI import Ui_MainWindow
from toolUI.TipsMessageBox import TipsMessageBox
from toolUI.cameranums import Camera

class DetectionThread(QThread):
    update_previous_signal = pyqtSignal(QImage)
    update_current_signal = pyqtSignal(QImage)

    def __init__(self, image_path, weights_path):
        super().__init__()
        self.image_path = image_path
        self.weights_path = weights_path
        self.model = YOLO(self.weights_path)

    def run(self):
        # 加载图像
        original_image = Image.open(self.image_path)
        # 创建图像副本进行检测
        image_to_detect = original_image.copy()
        results = self.model.predict(source=image_to_detect, save=False)

        # 将原始图像转换为QImage
        original_qimage = self.convert_pil_to_qimage(original_image)
        # 将检测后的图像转换为QImage
        detected_qimage = self.convert_pil_to_qimage(Image.fromarray(cv2.cvtColor(results.render()[0], cv2.COLOR_BGR2RGB)))

        # 发送信号更新GUI
        self.update_previous_signal.emit(original_qimage)
        self.update_current_signal.emit(detected_qimage)

    @staticmethod
    def convert_pil_to_qimage(pil_image):
        if pil_image.mode == "RGB":
            pass
        elif pil_image.mode == "L":
            pil_image = pil_image.convert("RGBA")
        data = pil_image.tobytes("raw", "BGRA")
        qimage = QImage(data, pil_image.size[0], pil_image.size[1], QImage.Format_ARGB32)
        return qimage

# 假设您的主窗口类如下
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.label_previous = QLabel(self)
        self.label_current = QLabel(self)
        # 设置布局和其他初始化...

        # 创建检测线程
        self.det_thread = DetectionThread(r'E:\boneage_pyqt5\test_img.jpg', r'E:\boneage_pyqt5\pt\yolov8n.pt')
        self.det_thread.update_previous_signal.connect(self.update_previous_label)
        self.det_thread.update_current_signal.connect(self.update_current_label)
        self.det_thread.start()

    def update_previous_label(self, qimage):
        self.label_previous.setPixmap(QPixmap.fromImage(qimage))

    def update_current_label(self, qimage):
        self.label_current.setPixmap(QPixmap.fromImage(qimage))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
