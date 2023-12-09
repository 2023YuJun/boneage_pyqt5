import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QGroupBox, QLabel, QWidget, QComboBox
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap, QImage

class CameraWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Camera Viewer")

        # 创建主布局和窗口部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # 创建摄像头选择下拉菜单
        self.camera_combo = QComboBox()
        self.camera_combo.addItem("Select Camera")
        self.detect_camera_devices()  # 检测摄像头设备
        self.layout.addWidget(self.camera_combo)

        # 创建按钮
        self.btn_toggle_camera = QPushButton("Start Camera")
        self.btn_toggle_camera.setCheckable(True)  # 设置按钮为可选中状态
        self.btn_toggle_camera.clicked.connect(self.toggle_camera)
        self.layout.addWidget(self.btn_toggle_camera)

        # 创建用于显示摄像头画面的 QLabel
        self.camera_groupbox = QGroupBox("Camera View")
        self.camera_layout = QVBoxLayout()
        self.camera_label = QLabel()
        self.camera_label.setFixedSize(640, 480)  # 设置显示画面的大小
        self.camera_layout.addWidget(self.camera_label)
        self.camera_groupbox.setLayout(self.camera_layout)
        self.layout.addWidget(self.camera_groupbox)

        # 初始化摄像头和计时器
        self.capture = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    def detect_camera_devices(self):
        for i in range(10):  # 检测摄像头设备，可根据实际情况调整范围
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.camera_combo.addItem(f"Camera {i}")
                cap.release()

    def toggle_camera(self):
        if not self.btn_toggle_camera.isChecked():
            self.stop_camera()
            self.camera_combo.setEnabled(True)
        else:
            self.start_camera()
            self.camera_combo.setEnabled(False)

    def start_camera(self):
        if self.capture is None:
            camera_index = self.camera_combo.currentIndex() - 1  # 获取选择的摄像头索引
            if camera_index >= 0:
                self.capture = cv2.VideoCapture(camera_index)  # 打开选择的摄像头
                self.timer.start(20)  # 设置更新画面的时间间隔
                self.btn_toggle_camera.setText("Stop Camera")

    def stop_camera(self):
        self.timer.stop()
        if self.capture is not None:
            self.capture.release()  # 释放摄像头
            self.capture = None  # 将 self.capture 设为 None
        self.camera_label.clear()
        self.btn_toggle_camera.setText("Start Camera")

    def update_frame(self):
        ret, frame = self.capture.read()  # 读取摄像头画面

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 将图像格式从 OpenCV 格式转换为 RGB 格式
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(convert_to_Qt_format)
            self.camera_label.setPixmap(pixmap.scaled(self.camera_label.size()))  # 在 QLabel 中显示画面

    def closeEvent(self, event):
        if self.capture is not None:
            self.capture.release()  # 释放摄像头
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = CameraWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
