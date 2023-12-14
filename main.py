import sys
import os
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os
import time
import cv2 

from PyQt5.QtCore import Qt, QPoint, QTimer, QThread, pyqtSignal,QSize
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu, QAction, QLabel, QVBoxLayout, QWidget,QSplitter,QSizePolicy
from PyQt5.QtGui import QImage, QPixmap, QPainter, QIcon,QCursor
from PyQt5 import uic
from mainUI.mainUI import Ui_MainWindow
from toolUI.TipsMessageBox import TipsMessageBox
from toolUI.cameranums import Camera


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
        self.cameraButton.clicked.connect(self.chose_cam)

        self.runButton.setCheckable(True)
        self.runButton.clicked.connect(self.toggle_video_play)

        # 初始化摄像头和计时器
        self.cam = '0'
        self.capture = None
        self.captimer = QTimer(self)
        self.captimer.timeout.connect(self.update_frame)

        # 视频播放
        self.video = None
        self.total_frames = 0
        self.current_frame = 0
        self.videotimer = QTimer(self)
        self.videotimer.timeout.connect(self.show_next_frame)

        # 更改读条
        self.confSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'confSpinBox'))
        self.confSlider.valueChanged.connect(lambda x: self.change_val(x, 'confSlider'))
        self.iouSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'iouSpinBox'))
        self.iouSlider.valueChanged.connect(lambda x: self.change_val(x, 'iouSlider'))
        self.rateSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'rateSpinBox'))
        self.rateSlider.valueChanged.connect(lambda x: self.change_val(x, 'rateSlider'))

        
        """ 
        TODO 
        1.最大化窗口后，分割器的宽高也要跟着变化,同时两个显示的组件的宽高也要跟着变化
        2.图片/视频自适应窗口大小
        """
        # 创建分割器实例
        self.splitter=QSplitter(Qt.Horizontal,self.groupBox_8)
        self.splitter.setEnabled(True)
        self.splitter.resize(899,426)
        self.splitter.setHandleWidth(10)
        self.splitter.setStyleSheet("QSplitter::handle { background-color: red; }")#设置分隔条的样式
        # 创建组件label_current
        self.label_current=QLabel()
        #  TEST 测试图片
        # pic1=QPixmap("./pic/b939.jpg")
        # self.label_current.setPixmap(pic1)
        self.label_current.setAlignment(Qt.AlignCenter)
        self.label_current.setScaledContents(True)
        sizePolicy = QSizePolicy(QSizePolicy.Ignored,QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_current.sizePolicy().hasHeightForWidth())
        self.label_current.setSizePolicy(sizePolicy)
        self.label_current.setMinimumSize(QSize(200, 0))
        self.label_current.setCursor(QCursor(Qt.ArrowCursor))

        # TEST 测试OpenCV打开的图片
        self.cvimg=cv2.imread("./pic/F0UyYDUWAAAuQux.png")
        # OpenCV图片转为QImage 再转为QPixmap
        height,width,depth=self.cvimg.shape
        img=QImage(self.cvimg.data,width,height,width*depth,QImage.Format_BGR888)
        self.label_current.setPixmap(QPixmap.fromImage(img))

        # TEST 测试打开摄像头(只是打开了摄像头，只有一帧的画面)
        flag,self.image = cv2.VideoCapture(0).read()
        show = cv2.resize(self.image,(480,320))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QImage(show.data, show.shape[1],show.shape[0],QImage.Format_RGB888)
        self.label_current.setPixmap(QPixmap.fromImage(showImage))


        # 创建组件label_previous
        self.label_previous=QLabel()
        # TEST 
        # pic2=QPixmap("./pic/F0UyYDUWAAAuQux.png")
        # self.label_previous.setPixmap(pic2)
        self.label_previous.setAlignment(Qt.AlignCenter)
        sizePolicy.setHeightForWidth(self.label_previous.sizePolicy().hasHeightForWidth())
        self.label_previous.setSizePolicy(sizePolicy)
        self.label_previous.setMinimumSize(QSize(200, 0))
        self.label_previous.setCursor(QCursor(Qt.ArrowCursor))

        # TEST 测试OpenCV打开的图片
        self.cvimg1=cv2.imread("./pic/b939.jpg")
        # OpenCV图片转为QImage 再转为QPixmap
        height,width,depth=self.cvimg1.shape
        img=QImage(self.cvimg1.data,width,height,width*depth,QImage.Format_BGR888)
        self.label_previous.setPixmap(QPixmap.fromImage(img))

        # 分割器添加组件
        self.splitter.addWidget(self.label_current)
        self.splitter.addWidget(self.label_previous)



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
        name, _ = QFileDialog.getOpenFileName(self, 'Video/image', open_fold, "video/image file(*.mp4 *.mkv *.avi *.flv"
                                                                              "*.jpg *.png)")
        if name:
            if name.lower().endswith(('.png', '.jpg', '.jpeg')):
                if self.video is not None:
                    self.videotimer.stop()
                    self.video.release()
                    self.video = None
                pixmap = QPixmap(name)
                self.label_previous.setPixmap(pixmap.scaled(self.label_previous.size(), Qt.KeepAspectRatio))
            else:
                self.show_video(name)
            self.bottom_msg('Loaded file：{}'.format(os.path.basename(name)))
            config['Default_path'] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)

    def show_video(self, file_path):
        self.video = cv2.VideoCapture(file_path)
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频总帧数
        self.videotimer.start(20)  # 设置计时器，根据视频帧率调整时间间隔
        self.current_frame = 0

    def show_next_frame(self):
        if self.video and self.video.isOpened():
            ret, frame = self.video.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                q_img = QPixmap.fromImage(QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888))
                self.label_previous.setPixmap(q_img.scaled(self.label_previous.size(), Qt.KeepAspectRatio))

                self.current_frame += 1
                progress_value = int((self.current_frame / self.total_frames) * 100)
                self.progressBar.setValue(progress_value)

                QApplication.processEvents()
            else:
                self.videotimer.stop()
                self.video.release()
                self.video = None

    def toggle_video_play(self):
        if self.runButton.isChecked():
            if not self.video:
                return
            else:
                self.videotimer.start(20)
        else:
            self.videotimer.stop()

    def change_val(self, x, flag):
        if flag == 'confSpinBox':
            self.confSlider.setValue(int(x * 100))
        elif flag == 'confSlider':
            self.confSpinBox.setValue(x / 100)
            # self.det_thread.conf_thres = x / 100
        elif flag == 'iouSpinBox':
            self.iouSlider.setValue(int(x * 100))
        elif flag == 'iouSlider':
            self.iouSpinBox.setValue(x / 100)
            # self.det_thread.iou_thres = x / 100
        elif flag == 'rateSpinBox':
            self.rateSlider.setValue(x)
        elif flag == 'rateSlider':
            self.rateSpinBox.setValue(x)
            # self.det_thread.rate = x * 10
        else:
            pass

    def chose_cam(self):
        if self.cameraButton.isChecked():
            try:
                TipsMessageBox(
                    self.closeButton, title='Tips', text='Loading camera', time=2000, auto=True).exec_()
                # get the number of local cameras
                _, cams = Camera().get_cam_num()
                popMenu = QMenu()
                popMenu.setFixedWidth(self.cameraButton.width())
                popMenu.setStyleSheet('''
                                                QMenu {
                                                font-size: 16px;
                                                font-family: "Microsoft YaHei UI";
                                                font-weight: light;
                                                color:white;
                                                border-style: solid;
                                                border-width: 0px;
                                                border-color: rgba(255, 255, 255, 255);
                                                border-radius: 3px;
                                                background-color: rgba(200, 200, 200,50);}
                                                ''')

                for cam in cams:
                    exec("action_%s = QAction('%s')" % (cam, cam))
                    exec("popMenu.addAction(action_%s)" % cam)

                x = self.groupBox_input.mapToGlobal(self.cameraButton.pos()).x()
                y = self.groupBox_input.mapToGlobal(self.cameraButton.pos()).y()
                y = y + self.cameraButton.frameGeometry().height()
                pos = QPoint(x, y)
                action = popMenu.exec_(pos)
                if action:
                    self.cam = action.text()
                    self.start_camera()
                    self.bottom_msg('Loading camera：{}'.format(action.text()))
            except Exception as e:
                self.bottom_msg('%s' % e)
        else:
            TipsMessageBox(
                self.closeButton, title='Tips', text='closing camera', time=2000, auto=True).exec_()
            self.stop_camera()
            self.bottom_msg('')

    def start_camera(self):
        if self.capture is None:
            camera_index = int(self.cam)  # 获取选择的摄像头索引
            if camera_index >= 0:
                self.capture = cv2.VideoCapture(camera_index)  # 打开选择的摄像头
                self.captimer.start(20)  # 设置更新画面的时间间隔

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

    def bottom_msg(self, msg):
        self.label_bottom.setText(msg)
        # self.qtimer.start(3000)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myForm = MainWindow()
    myForm.show()
    app.exec()
