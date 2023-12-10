from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QVBoxLayout, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
import cv2
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
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu, QAction, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap, QPainter, QIcon
from PyQt5 import uic
from mainUI.mainUI import Ui_MainWindow
from toolUI.TipsMessageBox import TipsMessageBox
from toolUI.cameranums import Camera

from PyQt5.QtWidgets import QMainWindow, QLabel, QApplication, QVBoxLayout, QPushButton, QWidget, QProgressBar
from PyQt5.QtCore import QTimer, Qt
import cv2
import sys

class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.video_label = QLabel(self)
        self.setCentralWidget(self.video_label)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setValue(0)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.show_video_frame)

        self.play_button = QPushButton('Play', self)
        self.play_button.clicked.connect(self.play_video)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.progress_bar)
        self.layout.addWidget(self.play_button)

        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)

        self.file_path = 'your_video_file.mp4'  # 你的视频文件路径
        self.cap = cv2.VideoCapture(self.file_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = 0

    def play_video(self):
        if not self.timer.isActive():
            self.timer.start(1000 // self.fps)  # 设置计时器，根据视频帧率调整时间间隔
            self.play_button.setText('Pause')
        else:
            self.timer.stop()
            self.play_button.setText('Play')

    def show_video_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            pixmap = self.convert_frame_to_pixmap(frame)
            self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio))
            progress_value = int((self.frame_count / self.total_frames) * 100)
            self.progress_bar.setValue(progress_value)
        else:
            self.cap.release()
            self.timer.stop()
            self.play_button.setText('Play')

    def convert_frame_to_pixmap(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        return pixmap


def main():
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.resize(800, 600)
    player.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
