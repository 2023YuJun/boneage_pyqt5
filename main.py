import sys
import json
import numpy as np
import os
import time
import cv2

from PyQt5.QtCore import Qt, QPoint, QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu, QAction, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap, QPainter, QIcon
from PyQt5 import uic, QtWidgets, QtGui, QtCore
from mainUI.mainUI import Ui_MainWindow
from toolUI.TipsMessageBox import TipsMessageBox
from toolUI.cameranums import Camera

from re_writ import CheckableComboBox, replace_comboBox_with_checkable

from ultralytics import YOLO
from PIL import Image


class DetThread(QThread):
    update_image = pyqtSignal(np.ndarray)
    progress_updated = pyqtSignal(int)

    def __init__(self):
        super(DetThread, self).__init__()
        self.source = None
        self.running = False
        self.paused = False
        self.total_frames = 0
        self.current_frame = 0
        self.cap = None

    def run(self):
        self.running = True

        if isinstance(self.source, str) and not self.source.lower().endswith(('.jpg', '.png')):
            if self.source.isdigit():
                self.cap = cv2.VideoCapture(int(self.source))
            else:
                self.cap = cv2.VideoCapture(self.source)

        if self.cap:
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while self.running:
            if isinstance(self.source, str) and self.source.lower().endswith(('.jpg', '.png')):
                frame = cv2.imread(self.source)
                if frame is not None:
                    self.update_image.emit(frame)
                break

            if not self.paused and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.update_image.emit(frame)
                    self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    self.progress_updated.emit(self.current_frame)
                    time.sleep(1 / self.cap.get(cv2.CAP_PROP_FPS))  # 控制视频播放速度
                else:
                    break
            else:
                time.sleep(0.1)  # 减少CPU使用率

        if self.cap:
            self.cap.release()

    def stop(self):
        self.running = False

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def set_position(self, frame_number):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            if ret:
                self.update_image.emit(frame)


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.model_type = None
        self.setupUi(self)
        self.m_flag = False

        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint
                            | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)

        self.maxButton.setCheckable(True)
        self.maxButton.clicked.connect(self.max_or_restore)
        self.maxButton.animateClick(10)

        self.comboBox_model = replace_comboBox_with_checkable(self.comboBox_model)
        self.comboBox_task.addItems(["骨龄评估", "检测", "分类"])
        self.model_list = []
        self.update_model_list()
        self.comboBox_task.currentTextChanged.connect(self.update_model_items)
        self.update_model_items()

        self.qtimer_search = QtCore.QTimer(self)
        self.qtimer_search.timeout.connect(self.update_model_list)
        self.qtimer_search.start(1000)

        self.fileButton.clicked.connect(self.open_file)

        self.cameraButton.setCheckable(True)
        self.cameraButton.clicked.connect(self.chose_cam)

        self.runButton.setCheckable(True)
        self.runButton.clicked.connect(self.run_or_continue)
        self.runButton.setChecked(False)  # 默认关闭状态

        self.confSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'confSpinBox'))
        self.confSlider.valueChanged.connect(lambda x: self.change_val(x, 'confSlider'))
        self.iouSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'iouSpinBox'))
        self.iouSlider.valueChanged.connect(lambda x: self.change_val(x, 'iouSlider'))
        self.rateSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'rateSpinBox'))
        self.rateSlider.valueChanged.connect(lambda x: self.change_val(x, 'rateSlider'))

        self.qtimer = QTimer(self)
        self.qtimer.setSingleShot(True)
        self.qtimer.timeout.connect(lambda: self.label_bottom.clear())

        self.checkBox_enable.clicked.connect(self.checkrate)
        self.checkBox_autosave.clicked.connect(self.is_save)
        self.load_setting()

        self.det_thread = DetThread()
        self.det_thread.update_image.connect(lambda img: self.show_image(img, self.label_previous))
        self.det_thread.progress_updated.connect(self.update_progress)

        self.ProgressSlider.sliderPressed.connect(self.on_slider_pressed)
        self.ProgressSlider.sliderReleased.connect(self.on_slider_released)
        self.ProgressSlider.sliderMoved.connect(self.on_slider_moved)

        self.preheat_models()

    def preheat_models(self):
        for model_file in self.model_list:
            model_path = os.path.join('./model', model_file)
            model = YOLO(model_path)  # 使用YOLO进行预热
            del model
        self.show_tips('Model preheating completed')

    def show_tips(self, text):
        TipsMessageBox(self.closeButton, title='Tips', text=text, time=2000, auto=True).exec_()

    def update_progress(self, frame_number):
        if self.det_thread.total_frames > 0:
            self.ProgressSlider.setValue(int((frame_number / self.det_thread.total_frames) * 100))

    def on_slider_pressed(self):
        self.det_thread.pause()
        self.runButton.setChecked(False)

    def on_slider_released(self):
        self.runButton.setChecked(True)
        self.det_thread.resume()

    def on_slider_moved(self, position):
        frame_number = int((position / 100) * self.det_thread.total_frames)
        self.det_thread.set_position(frame_number)

    def run_or_continue(self):
        if self.runButton.isChecked():
            self.det_thread.resume()
        else:
            self.det_thread.pause()

    def update_model_list(self):
        model_list = os.listdir('./model')
        model_list = [file for file in model_list if file.endswith('.pt')]
        model_list.sort(key=lambda x: os.path.getsize('./model/' + x))

        if model_list != self.model_list:
            self.model_list = model_list
            self.update_model_items()

    def update_model_items(self):
        selected_task = self.comboBox_task.currentText()
        self.comboBox_model.clear()

        if selected_task == "骨龄评估":
            cls_models = [file for file in self.model_list if file.endswith('-cls.pt')]
            det_models = [file for file in self.model_list if file.endswith('-det.pt')]
            self.comboBox_model.setMultiSelect(True)
            self.comboBox_model.addItems(cls_models + det_models)
        elif selected_task == "检测":
            filtered_list = [file for file in self.model_list if file.endswith('-det.pt')]
            self.comboBox_model.setMultiSelect(False)
            self.comboBox_model.addItems(filtered_list)
        elif selected_task == "分类":
            filtered_list = [file for file in self.model_list if file.endswith('-cls.pt')]
            self.comboBox_model.setMultiSelect(False)
            self.comboBox_model.addItems(filtered_list)

        self.comboBox_model.updateSelectedFiles()

    def open_file(self):
        config_file = 'config/Default_path.json'
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['Default_path']
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, 'Video/image', open_fold, "video/image file(*.mp4 *.mkv *.avi *.flv *.jpg *.png)")
        if name:
            self.det_thread.stop()
            self.det_thread.wait()
            self.label_previous.clear()
            self.det_thread.source = name
            self.ProgressSlider.setValue(0)
            self.det_thread.start()
            self.runButton.setChecked(True)
            self.bottom_msg('Loaded file：{}'.format(os.path.basename(name)))
            config['Default_path'] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)

    def chose_cam(self):
        if self.cameraButton.isChecked():
            try:
                TipsMessageBox(self.closeButton, title='Tips', text='Loading camera', time=2000, auto=True).exec_()
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
                    self.det_thread.stop()
                    self.det_thread.wait()
                    self.label_previous.clear()
                    self.det_thread.source = action.text()
                    self.ProgressSlider.setValue(0)
                    self.det_thread.start()
                    self.runButton.setChecked(True)
                    self.bottom_msg('Loading camera：{}'.format(action.text()))
            except Exception as e:
                self.bottom_msg('%s' % e)
        else:
            self.det_thread.stop()
            self.det_thread.wait()
            self.label_previous.clear()
            self.show_tips('Camera closed')

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

    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()

            if iw / w > ih / h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))
            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))
        except Exception as e:
            print(repr(e))

    def change_model(self, x):
        self.model_type = self.comboBox_model.currentText()
        self.bottom_msg('Change model to %s' % x)

    def load_setting(self):
        config_file = 'config/setting.json'
        if not os.path.exists(config_file):
            iou = 0.26
            conf = 0.33
            rate = 10
            check = 0
            savecheck = 0
            new_config = {"iou": iou,
                          "conf": conf,
                          "rate": rate,
                          "check": check,
                          "savecheck": savecheck
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            if len(config) != 5:
                iou = 0.26
                conf = 0.33
                rate = 10
                check = 0
                savecheck = 0
            else:
                iou = config['iou']
                conf = config['conf']
                rate = config['rate']
                check = config['check']
                savecheck = config['savecheck']
        self.confSpinBox.setValue(conf)
        self.iouSpinBox.setValue(iou)
        self.rateSpinBox.setValue(rate)
        self.checkBox_enable.setCheckState(check)
        self.checkBox_autosave.setCheckState(savecheck)
        self.is_save()

    def is_save(self):
        pass

    def checkrate(self):
        pass

    def stop(self):
        self.det_thread.stop()
        self.det_thread.wait()
        self.label_previous.clear()
        self.checkBox_autosave.setEnabled(True)

    def bottom_msg(self, msg):
        self.label_bottom.setText(msg)

    def show_msg(self, msg):
        self.runButton.setChecked(Qt.Unchecked)
        self.bottom_msg(msg)
        if msg == "Finished":
            self.checkBox_autosave.setEnabled(True)

    def show_statistic(self, statistic_dic):
        try:
            self.listWidget.clear()
            statistic_dic = sorted(statistic_dic.items(), key=lambda x: x[1], reverse=True)
            statistic_dic = [i for i in statistic_dic if i[1] > 0]
            results = [' ' + str(i[0]) + '：' + str(i[1]) for i in statistic_dic]
            self.listWidget.addItems(results)
        except Exception as e:
            print(repr(e))

    def max_or_restore(self):
        if self.maxButton.isChecked():
            self.window().showMaximized()
        else:
            self.window().showNormal()

    def mousePressEvent(self, event):
        self.m_Position = event.pos()
        if event.button() == Qt.LeftButton:
            if 0 < self.m_Position.x() < self.groupBox.pos().x() + self.groupBox.width() and \
                    0 < self.m_Position.y() < self.groupBox.pos().y() + self.groupBox.height():
                self.m_flag = True

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False

    def closeEvent(self, event):
        config_file = 'config/setting.json'
        config = dict()
        config['iou'] = self.iouSpinBox.value()
        config['conf'] = self.confSpinBox.value()
        config['rate'] = self.rateSpinBox.value()
        config['check'] = self.checkBox_enable.checkState()
        config['savecheck'] = self.checkBox_autosave.checkState()
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)
        TipsMessageBox(self.closeButton, title='Tips', text='Closing the program', time=2000, auto=True).exec_()
        sys.exit(0)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myForm = MainWindow()
    myForm.show()
    app.exec()
