import time
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

import common
from arthrosis_classify import process_images, cal_boneage
from arthrosis_detection import process
from common import detection_info, classify_info


class DetThread(QThread):
    """
    线程类，负责处理检测任务并更新UI界面。

    Attributes:
        update_label_previous (pyqtSignal): 信号，用于更新前一帧图像。
        update_label_current (pyqtSignal): 信号，用于更新当前帧图像。
        progress_updated (pyqtSignal): 信号，用于更新进度条。
        update_fps (pyqtSignal): 信号，用于更新FPS显示。
        update_bottom_info (pyqtSignal): 信号，用于更新底部信息。
        update_report_info (pyqtSignal): 信号，用于更新报告信息。
        update_detection_info (pyqtSignal): 信号，用于更新检测信息。
    """

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
        self.cap = self.initialize_capture()

        if self.cap:
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while self.running:
            if self.process_image_source():
                break

            if not self.paused and self.cap and self.cap.isOpened():
                self.process_video_frame()
            else:
                time.sleep(0.1)

        if self.cap:
            self.cap.release()

    def initialize_capture(self):
        """
        初始化视频捕捉器。
        """
        if isinstance(self.source, str) and not self.source.lower().endswith(('.jpg', '.png')):
            if self.source.isdigit():
                return cv2.VideoCapture(int(self.source))
            return cv2.VideoCapture(self.source)
        return None

    def process_image_source(self):
        """
        处理图像源。
        """
        if isinstance(self.source, str) and self.source.lower().endswith(('.jpg', '.png')):
            frame = cv2.imread(self.source)
            if frame is not None:
                self.update_label_previous.emit(frame)
                if self.detect_button_status:
                    self.process_frame(frame)
                else:
                    self.update_label_current.emit(frame)
            return True
        return False

    def process_video_frame(self):
        """
        处理视频帧。
        """
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
            self.running = False

    def process_frame(self, frame):
        """
        处理单帧图像。
        """
        detection_model, classify_model = self.load_models()
        common.classify_info.clear()
        common.detection_info.clear()
        common.REPORT = ""

        if self.task == "骨龄评估":
            self.process_bone_age_assessment(detection_model, classify_model, frame)
        elif self.task == "检测":
            self.process_detection(detection_model, frame)
        elif self.task == "分类":
            self.process_classification(classify_model, frame)

    def load_models(self):
        """
        加载检测和分类模型。
        """
        detection_model = None
        classify_model = None
        for model_path in self.model_paths:
            if '-det.pt' in model_path:
                detection_model = self.models[model_path]
            elif '-cls.pt' in model_path:
                classify_model = self.models[model_path]
        return detection_model, classify_model

    def process_bone_age_assessment(self, detection_model, classify_model, frame):
        """
        处理骨龄评估任务。
        """
        if detection_model and classify_model:
            detection_results, cropped_images = process(detection_model, [frame], iou=self.iou, conf=self.conf,
                                                        only_detect=False)
            for processed_frame in detection_results:
                self.update_label_current.emit(processed_frame)

            classify_results = process_images(classify_model, cropped_images, iou=self.iou, conf=self.conf)
            self.update_detection_info.emit('\n'.join(detection_info) + '\n' + '\n'.join(classify_info))

            sex = 'boy' if self.is_male else 'girl'
            bone_age = cal_boneage(sex, classify_results)
            self.update_report_info.emit(common.REPORT)

    def process_detection(self, detection_model, frame):
        """
        处理检测任务。
        """
        if detection_model:
            detection_results, _ = process(detection_model, [frame], iou=self.iou, conf=self.conf, only_detect=True)
            for processed_frame in detection_results:
                self.update_label_current.emit(processed_frame)
            self.update_detection_info.emit('\n'.join(detection_info))

    def process_classification(self, classify_model, frame):
        """
        处理分类任务。
        """
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