import sys
import json
import numpy as np
import os
import time
import cv2

from PyQt5.QtCore import Qt, QPoint, QTimer, QThread, pyqtSignal, QObject
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu, QAction
from PyQt5.QtGui import QImage, QPixmap
from mainUI.mainUI import Ui_MainWindow
from toolUI.TipsMessageBox import TipsMessageBox
from toolUI.cameranums import Camera

from toolUI.CheckableComboBox import replace_comboBox_with_checkable
from arthrosis_detection import process
from arthrosis_classify import process_images, cal_boneage
from common import detection_info, classify_info
import common

from ultralytics import YOLO


class DetThread(QThread):
    update_label_previous = pyqtSignal(np.ndarray)
    update_label_current = pyqtSignal(np.ndarray)
    progress_updated = pyqtSignal(int)
    update_fps = pyqtSignal(float)
    update_bottom_info = pyqtSignal(str)
    update_report_info = pyqtSignal(str)
    update_detection_info = pyqtSignal(str)

    def __init__(self):
        super(DetThread, self).__init__()
        self.source = None
        self.running = False
        self.paused = False
        self.total_frames = 0
        self.current_frame = 0
        self.cap = None
        self.conf = 0.50
        self.iou = 0.20
        self.rate = 1
        self.model_paths = []
        self.models = {}
        self.is_male = True
        self.task = None
        self.detect_button_status = False

    def run(self):
        self.running = True

        if isinstance(self.source, str) and not self.source.lower().endswith(('.jpg', '.png')):
            if self.source.isdigit():
                self.cap = cv2.VideoCapture(int(self.source))
            else:
                self.cap = cv2.VideoCapture(self.source)
        else:
            self.cap = None

        if self.cap:
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while self.running:
            if isinstance(self.source, str) and self.source.lower().endswith(('.jpg', '.png')):
                frame = cv2.imread(self.source)
                if frame is not None:
                    self.update_label_previous.emit(frame)
                    if self.detect_button_status:
                        self.process_frame(frame)
                    else:
                        self.update_label_current.emit(frame)
                break

            if not self.paused and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.update_label_previous.emit(frame)
                    self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    self.progress_updated.emit(self.current_frame)

                    if self.detect_button_status:
                        self.process_frame(frame)
                    else:
                        self.update_label_current.emit(frame)

                    fps = self.cap.get(cv2.CAP_PROP_FPS) + 10
                    self.update_fps.emit(fps - self.rate)

                    sleep_time = (1 / (fps - self.rate))
                    time.sleep(sleep_time if sleep_time > 0 else 0.01)
                else:
                    break
            else:
                time.sleep(0.1)

        if self.cap:
            self.cap.release()

    def process_frame(self, frame):
        detection_model = None
        classify_model = None
        common.classify_info.clear()
        common.detection_info.clear()
        common.REPORT = ""
        if self.task == "骨龄评估":
            for model_path in self.model_paths:
                if '-det.pt' in model_path:
                    detection_model = self.models[model_path]
                elif '-cls.pt' in model_path:
                    classify_model = self.models[model_path]

            if detection_model and classify_model:
                detection_results, cropped_images = process(detection_model, [frame],
                                                            iou=self.iou, conf=self.conf, only_detect=False)
                for processed_frame in detection_results:
                    self.update_label_current.emit(processed_frame)

                classify_results = process_images(classify_model, cropped_images, iou=self.iou, conf=self.conf)
                self.update_detection_info.emit('\n'.join(detection_info) + '\n' + '\n'.join(classify_info))

                sex = 'boy' if self.is_male else 'girl'
                bone_age = cal_boneage(sex, classify_results)
                self.update_report_info.emit(common.REPORT)

        elif self.task == "检测":
            for model_path in self.model_paths:
                if '-det.pt' in model_path:
                    detection_model = self.models[model_path]

            if detection_model:
                detection_results, _ = process(detection_model, [frame],
                                               iou=self.iou, conf=self.conf, only_detect=True)
                for processed_frame in detection_results:
                    self.update_label_current.emit(processed_frame)
                self.update_detection_info.emit('\n'.join(detection_info))

        elif self.task == "分类":
            for model_path in self.model_paths:
                if '-cls.pt' in model_path:
                    classify_model = self.models[model_path]

            if classify_model:
                classify_results = process_images(classify_model, [frame], iou=self.iou, conf=self.conf)
                for processed_frame in classify_results:
                    self.update_label_current.emit(processed_frame)
                self.update_detection_info.emit('\n'.join(classify_info))

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
                self.update_label_previous.emit(frame)


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.m_flag = False

        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint
                            | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)

        self.det_thread = DetThread()
        self.det_thread.update_label_current.connect(lambda img: self.show_image(img, self.label_current))
        self.det_thread.update_label_previous.connect(lambda img: self.show_image(img, self.label_previous))
        self.det_thread.progress_updated.connect(self.update_progress)
        self.det_thread.update_fps.connect(self.update_fps_label)
        self.det_thread.update_bottom_info.connect(self.bottom_msg)  # 连接新信号
        self.det_thread.update_report_info.connect(self.report_info.setPlainText)  # 连接新信号
        self.det_thread.update_detection_info.connect(self.detection_info.setPlainText)  # 连接新信号

        self.maxButton.setCheckable(True)
        self.maxButton.clicked.connect(self.max_or_restore)
        self.maxButton.animateClick(10)

        self.comboBox_model = replace_comboBox_with_checkable(self.comboBox_model)
        self.comboBox_task.addItems(["骨龄评估", "检测", "分类"])
        self.models = {}
        self.model_list = []
        self.update_model_list()
        self.comboBox_task.currentTextChanged.connect(self.update_model_items)
        self.comboBox_model.currentTextChanged.connect(self.select_model)

        self.fileButton.clicked.connect(self.open_file)

        self.cameraButton.clicked.connect(self.chose_cam)

        self.runButton.clicked.connect(self.run_or_continue)
        self.runButton.setChecked(False)

        self.detectButton.clicked.connect(self.detect)

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
        self.checkBox_male.toggled.connect(self.gender_checked)
        self.checkBox_female.toggled.connect(self.gender_checked)

        self.ProgressSlider.sliderPressed.connect(self.on_slider_pressed)
        self.ProgressSlider.sliderReleased.connect(self.on_slider_released)
        self.ProgressSlider.sliderMoved.connect(self.on_slider_moved)

        self.load_setting()
        self.preheat_models()

    def preheat_models(self):
        for model_file in self.model_list:
            model_path = os.path.join('./model', model_file)
            self.models[model_file] = YOLO(model_path)  # 使用YOLO进行预热
        self.show_tips('Model preheating completed')

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

    def select_model(self, model_name):
        self.bottom_msg('Select model to %s' % model_name)

    def open_file(self):
        config_file = 'config/Default_path.json'
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['Default_path']
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, 'Video/image', open_fold,
                                              "video/image file(*.mp4 *.mkv *.avi *.flv *.jpg *.png)")
        try:
            if name:
                self.det_thread.stop()
                self.det_thread.wait()
                self.label_previous.clear()
                self.label_current.clear()
                self.det_thread.source = name
                self.ProgressSlider.setValue(0)
                self.det_thread.start()
                self.runButton.setChecked(True)
                self.bottom_msg('Loaded file：{}'.format(os.path.basename(name)))
                config['Default_path'] = os.path.dirname(name)
                config_json = json.dumps(config, ensure_ascii=False, indent=2)
                with open(config_file, 'w', encoding='utf-8') as f:
                    f.write(config_json)
                # self.detectButton.click()  # 获得资源直接开始检测
        except Exception as e:
            self.bottom_msg('%s' % e)

    def chose_cam(self):
        if self.cameraButton.isChecked():
            try:
                self.show_tips('Loading camera')
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
                    action = QAction(str(cam), self)
                    popMenu.addAction(action)

                x = self.groupBox_input.mapToGlobal(self.cameraButton.pos()).x()
                y = self.groupBox_input.mapToGlobal(self.cameraButton.pos()).y()
                y = y + self.cameraButton.frameGeometry().height()
                pos = QPoint(x, y)
                action = popMenu.exec_(pos)
                if action:
                    self.det_thread.stop()
                    self.det_thread.wait()
                    self.label_previous.clear()
                    self.label_current.clear()
                    self.det_thread.source = action.text()
                    self.ProgressSlider.setValue(0)
                    self.det_thread.start()
                    self.runButton.setChecked(True)
                    self.bottom_msg('Loading camera：{}'.format(action.text()))
                    # self.detectButton.click()  # 获得资源直接开始检测
            except Exception as e:
                self.bottom_msg('%s' % e)
        else:
            self.det_thread.stop()
            self.det_thread.wait()
            self.label_previous.clear()
            self.label_current.clear()
            self.detection_info.clear()
            self.report_info.clear()
            self.show_tips('Camera closing')

    def change_val(self, x, flag):
        if flag == 'confSpinBox':
            self.confSlider.setValue(int(x * 100))
        elif flag == 'confSlider':
            self.confSpinBox.setValue(x / 100)
            self.det_thread.conf = x / 100
        elif flag == 'iouSpinBox':
            self.iouSlider.setValue(int(x * 100))
        elif flag == 'iouSlider':
            self.iouSpinBox.setValue(x / 100)
            self.det_thread.iou = x / 100
        elif flag == 'rateSpinBox':
            self.rateSlider.setValue(x)
        elif flag == 'rateSlider':
            self.rateSpinBox.setValue(x)
            if self.checkBox_enable.isChecked():
                self.det_thread.rate = x
        else:
            pass

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

    def detect(self):
        self.detection_info.clear()
        self.report_info.clear()
        common.classify_info.clear()
        common.detection_info.clear()

        self.det_thread.detect_button_status = self.detectButton.isChecked()

        selected_models = self.comboBox_model.selected_items

        if not selected_models:
            selected_task = self.comboBox_task.currentText()
            if selected_task == "骨龄评估":
                cls_models = [file for file in self.model_list if file.endswith('-cls.pt')]
                det_models = [file for file in self.model_list if file.endswith('-det.pt')]
                if cls_models:
                    selected_models.append(cls_models[0])
                if det_models:
                    selected_models.append(det_models[0])
            else:
                filtered_list = [file for file in self.model_list if '-n' in file]
                if filtered_list:
                    selected_models.append(filtered_list[0])

            self.comboBox_model.selected_items = selected_models

        self.det_thread.model_paths = selected_models
        self.det_thread.models = {path: self.models[path] for path in selected_models}
        self.det_thread.is_male = self.checkBox_male.isChecked()
        self.det_thread.task = self.comboBox_task.currentText()

        if not self.det_thread.isRunning():
            self.det_thread.start()

    def checkrate(self):
        if self.checkBox_enable.isChecked():
            self.det_thread.rate = self.rateSpinBox.value()
        else:
            self.det_thread.rate = 0  # 恢复正常播放速度

    def gender_checked(self):
        if self.sender() == self.checkBox_male and self.checkBox_male.isChecked():
            self.checkBox_female.setChecked(False)
        elif self.sender() == self.checkBox_female and self.checkBox_female.isChecked():
            self.checkBox_male.setChecked(False)
        self.save_setting()

    def update_fps_label(self, fps):
        self.fps_label.setText(f"FPS: {fps:.2f}")

    def stop(self):
        self.det_thread.stop()
        self.det_thread.wait()
        self.label_previous.clear()
        self.checkBox_autosave.setEnabled(True)

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

    def load_setting(self):
        config_file = 'config/setting.json'
        if not os.path.exists(config_file):
            iou = 0.20
            conf = 0.50
            rate = 1
            check = 0
            male_checked = True
            female_checked = False
            new_config = {
                "iou": iou,
                "conf": conf,
                "rate": rate,
                "check": check,
                "male_checked": male_checked,
                "female_checked": female_checked
            }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            if len(config) != 6:
                iou = 0.20
                conf = 0.50
                rate = 1
                check = 0
                male_checked = True
                female_checked = False
            else:
                iou = config['iou']
                conf = config['conf']
                rate = config['rate']
                check = config['check']
                male_checked = config['male_checked']
                female_checked = config['female_checked']
        self.confSpinBox.setValue(conf)
        self.iouSpinBox.setValue(iou)
        self.rateSpinBox.setValue(rate)
        self.checkBox_enable.setCheckState(check)
        self.checkBox_male.setChecked(male_checked)
        self.checkBox_female.setChecked(female_checked)

    def show_tips(self, text):
        TipsMessageBox(self.closeButton, title='Tips', text=text, time=2000, auto=True).exec_()

    def bottom_msg(self, msg):
        self.label_bottom.setText(msg)

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

    def save_setting(self):
        config_file = 'config/setting.json'
        config = dict()
        config['iou'] = self.iouSpinBox.value()
        config['conf'] = self.confSpinBox.value()
        config['rate'] = self.rateSpinBox.value()
        config['check'] = self.checkBox_enable.checkState()
        config['male_checked'] = self.checkBox_male.isChecked()
        config['female_checked'] = self.checkBox_female.isChecked()
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)

    def closeEvent(self, event):
        self.save_setting()
        self.show_tips('Closing the program')
        sys.exit(0)


# 自定义 stdout 重定向类
class EmittingStream(QObject):
    textWritten = pyqtSignal(str)

    def write(self, text):
        if text.strip():  # 确保不是空行
            self.textWritten.emit(str(text))

    def flush(self):
        pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myForm = MainWindow()

    # 重定向 stdout 到 GUI
    sys.stdout = EmittingStream()
    sys.stdout.textWritten.connect(myForm.bottom_msg)

    myForm.show()
    app.exec()
