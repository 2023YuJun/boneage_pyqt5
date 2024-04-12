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

from ultralytics import YOLO
from PIL import Image
from ultralytics.utils.torch_utils import select_device
from ultralytics.utils.checks import check_imgsz, check_imshow
from ultralytics.utils.ops import non_max_suppression, scale_coords
from ultralytics.utils.plotting import colors
from ultralytics.data.loaders import LoadStreams, LoadImagesAndVideos
from ultralytics.data.annotator import auto_annotate


class DetThread(QThread):
    # 检测线程
    send_current = pyqtSignal(np.ndarray)  # 定义一个信号，用于发送图像数据
    send_previous = pyqtSignal(np.ndarray)  # 定义一个信号，用于发送原始图像数据
    send_statistic = pyqtSignal(dict)  # 定义一个信号，用于发送统计信息（字典形式）
    send_msg = pyqtSignal(str)  # 定义一个信号，用于发送消息文本
    send_percent = pyqtSignal(int)  # 定义一个信号，用于发送进度百分比
    send_fps = pyqtSignal(str)  # 定义一个信号，用于发送帧率信息

    def __init__(self):
        super(DetThread, self).__init__()
        self.weights = './pt/yolov8n.pt'  # 默认权重文件路径
        self.current_weight = './pt/yolov8n.pt'  # 当前权重文件路径
        self.source = '0'  # 数据源，默认为摄像头
        self.conf_thres = 0.25  # 置信度阈值
        self.iou_thres = 0.45  # IoU 阈值
        self.jump_out = False  # 控制是否跳出检测循环的标志
        self.is_continue = True  # 控制是否继续检测的标志
        self.percent_length = 1000  # 进度条的长度
        self.rate_check = True  # 是否启用延迟（控制帧率）
        self.rate = 100  # 检测的帧率
        self.save_fold = './result'  # 结果保存路径


    @torch.no_grad()
    def run(self,
            imgsz=640,  # 推理时的图像尺寸（像素）
            max_det=1000,  # 每张图像的最大检测数目
            device='',  # 设备选择，例如 cuda 设备可以是 0 或者 0,1,2,3，如果是 cpu 则为空字符串
            view_img=True,  # 是否显示结果图像
            save_txt=False,  # 是否保存结果到文本文件
            save_conf=False,  # 是否保存置信度到 --save-txt 标签
            save_crop=False,  # 是否保存裁剪的预测框
            nosave=False,  # 是否禁止保存图像或视频
            classes=None,  # 按类别过滤，例如 --class 0，或者 --class 0 2 3
            agnostic_nms=False,  # 类别无关 NMS
            augment=False,  # 增强推理
            visualize=False,  # 可视化特征
            update=False,  # 更新所有模型
            project='runs/detect',  # 保存结果到的项目路径
            name='exp',  # 保存结果的名称
            exist_ok=False,  # 存在的项目或名称是否可用，如果是则不递增
            line_thickness=3,  # 边框的厚度（像素）
            hide_labels=False,  # 是否隐藏标签
            hide_conf=False,  # 是否隐藏置信度
            half=False,  # 使用 FP16 半精度推理
            ):
        # 初始化
        try:
            # 选择计算设备
            device = select_device(device)
            # 若半精度(half)推理，且设备非 CPU，则设置 half 为 True
            half &= device.type != 'cpu'

            # 加载模型
            model = YOLO(self.weights, task="detect")
            num_params = 0
            # 统计模型参数数量
            for param in model.parameters():
                num_params += param.numel()
            stride = int(model.stride.max())  # 模型步长
            # 检查输入图像尺寸是否合适
            imgsz = check_imgsz(imgsz, stride=stride)
            # 获取类别名称
            names = model.module.names if hasattr(model, 'module') else model.names
            # 若使用半精度，则将模型转换为 FP16
            if half:
                model.half()

            # 数据加载器
            if self.source.isnumeric() or self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
                # 如果数据源是数字或以指定协议开头的 URL，使用网络摄像头
                view_img = check_imshow()
                cudnn.benchmark = True  # 设置为 True 可以加速常量图像尺寸的推理
                dataset = LoadStreams(self.source, vid_stride=stride)
            else:
                # 否则，使用本地图像或视频
                dataset = LoadImagesAndVideos(self.source, vid_stride=stride)

            # 运行推理
            if device.type != 'cpu':
                # 仅在设备非 CPU 时运行一次模型，用于初始化
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
            count = 0
            jump_count = 0
            start_time = time.time()
            dataset = iter(dataset)

            while True:
                # 中途退出检测循环
                if self.jump_out:
                    self.vid_cap.release()
                    self.send_percent.emit(0)
                    self.send_msg.emit('Stop')
                    if hasattr(self, 'out'):
                        self.out.release()
                    break
                # 切换模型
                if self.current_weight != self.weights:
                    # 重新加载模型
                    model = YOLO(self.weights, task="detect")
                    num_params = 0
                    for param in model.parameters():
                        num_params += param.numel()
                    stride = int(model.stride.max())  # 模型步长
                    imgsz = check_imgsz(imgsz, stride=stride)  # 检查图像尺寸
                    names = model.module.names if hasattr(model, 'module') else model.names  # 获取类别名称
                    if half:
                        model.half()  # 转换为 FP16
                    # 运行推理
                    if device.type != 'cpu':
                        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # 运行一次
                    self.current_weight = self.weights
                if self.is_continue:
                    # 获取图像信息
                    path, img, im0s, self.vid_cap = next(dataset)
                    count += 1
                    if count % 30 == 0 and count >= 30:
                        # 计算并发送每秒帧数
                        fps = int(30 / (time.time() - start_time))
                        self.send_fps.emit('fps：' + str(fps))
                        start_time = time.time()
                    if self.vid_cap:
                        # 计算并发送进度百分比
                        percent = int(count / self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT) * self.percent_length)
                        self.send_percent.emit(percent)
                    else:
                        percent = self.percent_length

                    # 统计信息字典初始化
                    statistic_dic = {name: 0 for name in names}
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 转换为 fp16/32
                    img /= 255.0  # 将像素值缩放到 [0, 1]
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    pred = model(img, augment=augment)[0]

                    # 应用 NMS
                    pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes, agnostic_nms,
                                               max_det=max_det)

                    # 处理检测结果
                    for i, det in enumerate(pred):  # 每张图像的检测结果
                        im0 = im0s.copy()
                        annotator = auto_annotate(im0, det_model=model)
                        if len(det):
                            # 将边界框从 img_size 转换为 im0 尺寸
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                            # 写入结果
                            for *xyxy, conf, cls in reversed(det):
                                c = int(cls)  # 类别索引
                                statistic_dic[names[c]] += 1
                                label = None if hide_labels else (
                                    names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                annotator.box_label(xyxy, label, color=colors(c, True))

                    # 是否启用帧率限制
                    if self.rate_check:
                        time.sleep(1 / self.rate)
                    im0 = annotator.result()
                    self.send_img.emit(im0)
                    self.send_raw.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])
                    self.send_statistic.emit(statistic_dic)

                    # 是否保存结果图像
                    if self.save_fold:
                        os.makedirs(self.save_fold, exist_ok=True)
                        if self.vid_cap is None:
                            save_path = os.path.join(self.save_fold,
                                                     time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime()) + '.jpg')
                            cv2.imwrite(save_path, im0)
                        else:
                            if count == 1:
                                ori_fps = int(self.vid_cap.get(cv2.CAP_PROP_FPS))
                                if ori_fps == 0:
                                    ori_fps = 25
                                width, height = im0.shape[1], im0.shape[0]
                                save_path = os.path.join(self.save_fold,
                                                         time.strftime('%Y_%m_%d_%H_%M_%S',
                                                                       time.localtime()) + '.mp4')
                                self.out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), ori_fps,
                                                           (width, height))
                            self.out.write(im0)

                    # 是否达到预定百分比
                    if percent == self.percent_length:
                        print(count)
                        self.send_percent.emit(0)
                        self.send_msg.emit('finished')
                        if hasattr(self, 'out'):
                            self.out.release()
                        break
        except Exception as e:
            # 发送异常信息
            self.send_msg.emit('%s' % e)


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
        self.runButton.clicked.connect(self.run_or_continue)
        self.stopButton.clicked.connect(self.stop)

        # yolov8 thread
        self.det_thread = DetThread()
        self.model_type = self.comboBox_model.currentText()
        self.det_thread.weights = "./pt/%s" % self.model_type
        self.det_thread.source = '0'
        self.det_thread.percent_length = self.progressBar.maximum()
        self.det_thread.send_previous.connect(lambda x: self.show_image(x, self.label_previous))
        self.det_thread.send_current.connect(lambda x: self.show_image(x, self.label_current))
        self.det_thread.send_statistic.connect(self.show_statistic)
        self.det_thread.send_msg.connect(lambda x: self.show_msg(x))
        self.det_thread.send_percent.connect(lambda x: self.progressBar.setValue(x))
        self.det_thread.send_fps.connect(lambda x: self.fps_label.setText(x))

        # # 初始化摄像头和计时器
        # self.cam = '0'
        # self.capture = None
        # self.captimer = QTimer(self)
        # self.captimer.timeout.connect(self.update_frame)

        # # 视频播放
        # self.video = None
        # self.total_frames = 0
        # self.current_frame = 0
        # self.videotimer = QTimer(self)
        # self.videotimer.timeout.connect(self.show_next_frame)

        # 更改读条
        self.confSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'confSpinBox'))
        self.confSlider.valueChanged.connect(lambda x: self.change_val(x, 'confSlider'))
        self.iouSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'iouSpinBox'))
        self.iouSlider.valueChanged.connect(lambda x: self.change_val(x, 'iouSlider'))
        self.rateSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'rateSpinBox'))
        self.rateSlider.valueChanged.connect(lambda x: self.change_val(x, 'rateSlider'))

        self.qtimer = QTimer(self)
        self.qtimer.setSingleShot(True)
        self.qtimer.timeout.connect(lambda: self.label_bottom.clear())

        # 自动更新模型
        self.comboBox_task.clear()
        self.comboBox_task.addItems(["骨龄评估", "检测", "分类"])
        self.comboBox_model.clear()
        self.pt_list = os.listdir('./pt')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize('./pt/' + x))
        self.comboBox_task.currentTextChanged.connect(self.update_model_items)
        self.update_model_items()
        self.comboBox_model.currentTextChanged.connect(self.change_model)
        self.qtimer_search = QTimer(self)
        self.qtimer_search.timeout.connect(lambda: self.search_pt())
        self.qtimer_search.start(1000)

        self.checkBox_enable.clicked.connect(self.checkrate)
        self.checkBox_autosave.clicked.connect(self.is_save)
        self.load_setting()

    def open_file(self):
        config_file = 'config/Default_path.json'
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['Default_path']
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, 'Video/image', open_fold, "video/image file(*.mp4 *.mkv *.avi *.flv"
                                                                              "*.jpg *.png)")
        if name:
            self.det_thread.source = name
            self.bottom_msg('Loaded file：{}'.format(os.path.basename(name)))
            config['Default_path'] = os.path.dirname(name)  # 保存此次访问文件路径
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            self.stop()

    # def show_video(self, file_path):
    #     self.video = cv2.VideoCapture(file_path)
    #     self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频总帧数
    #     self.videotimer.start(20)  # 设置计时器，根据视频帧率调整时间间隔
    #     self.current_frame = 0
    #
    #     # 触发一次 runButton 的点击
    #     self.runButton.click()

    # # 处理下一帧
    # def show_next_frame(self):
    #     if self.video and self.video.isOpened():
    #         ret, frame = self.video.read()
    #         if ret:
    #             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #             h, w, ch = frame.shape
    #             bytes_per_line = ch * w
    #             q_img = QPixmap.fromImage(QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888))
    #             self.label_previous.setPixmap(q_img.scaled(self.label_previous.size(), Qt.KeepAspectRatio))
    #
    #             self.current_frame += 1
    #             progress_value = int((self.current_frame / self.total_frames) * 100)
    #             self.progressBar.setValue(progress_value)
    #
    #             QApplication.processEvents()
    #         else:
    #             self.videotimer.stop()
    #             self.video.release()
    #             self.video = None

    # 视频播放暂停
    # def toggle_video_play(self):
    #     if self.runButton.isChecked():
    #         if not self.video:
    #             return
    #         else:
    #             self.videotimer.start(20)
    #     else:
    #         self.videotimer.stop()

    # 根据滑块或者SpinBox的值更新界面参数

    def chose_cam(self):
        # if self.cameraButton.isChecked():
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
                # self.cam = action.text()
                # self.start_camera()
                self.det_thread.source = action.text()
                self.bottom_msg('Loading camera：{}'.format(action.text()))
        except Exception as e:
            self.bottom_msg('%s' % e)

        # else:
        #     TipsMessageBox(
        #         self.closeButton, title='Tips', text='closing camera', time=2000, auto=True).exec_()
        #     self.stop_camera()
        #     self.bottom_msg('')

    def change_val(self, x, flag):
        if flag == 'confSpinBox':
            self.confSlider.setValue(int(x * 100))
        elif flag == 'confSlider':
            self.confSpinBox.setValue(x / 100)
            self.det_thread.conf_thres = x / 100
        elif flag == 'iouSpinBox':
            self.iouSlider.setValue(int(x * 100))
        elif flag == 'iouSlider':
            self.iouSpinBox.setValue(x / 100)
            self.det_thread.iou_thres = x / 100
        elif flag == 'rateSpinBox':
            self.rateSlider.setValue(x)
        elif flag == 'rateSlider':
            self.rateSpinBox.setValue(x)
            self.det_thread.rate = x * 10
        else:
            pass

    # def start_camera(self):
    #     if self.capture is None:
    #         camera_index = int(self.cam)  # 获取选择的摄像头索引
    #         if camera_index >= 0:
    #             self.capture = cv2.VideoCapture(camera_index)  # 打开选择的摄像头
    #             self.captimer.start(20)  # 设置更新画面的时间间隔

    # def stop_camera(self):
    #     if self.capture is not None:
    #         self.captimer.stop()  # 停止定时器
    #         self.capture.release()  # 释放摄像头
    #         self.capture = None  # 将 self.capture 设为 None
    #     self.label_previous.clear()

    # def update_frame(self):
    #     ret, frame = self.capture.read()  # 读取摄像头画面
    #     if ret:
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 将图像格式从 OpenCV 格式转换为 RGB 格式
    #         h, w, ch = frame.shape
    #         bytes_per_line = ch * w
    #         convert_to_Qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
    #         pixmap = QPixmap.fromImage(convert_to_Qt_format)
    #         self.label_previous.setPixmap(pixmap.scaled(self.label_previous.size()))  # 在 QLabel 中显示画面

    # 显示图像在指定的 QLabel 中
    @staticmethod
    def show_image(img_src, label):
        try:
            # 获取图像的高度、宽度和通道数（对于 RGB 图像通道数为 3）
            ih, iw, _ = img_src.shape

            # 获取 QLabel 的宽度和高度
            w = label.geometry().width()
            h = label.geometry().height()

            # 保持原始图像的纵横比
            if iw / w > ih / h:  # 如果图像宽度比 QLabel 宽度大，按宽度缩放
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))  # 调整图像大小

            else:  # 如果图像高度比 QLabel 高度大，按高度缩放
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))  # 调整图像大小

            # 将图像从 OpenCV 格式转换为 Qt 的 QImage 格式
            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)

            # 将 QImage 设置为 QLabel 的 pixmap，以在界面上显示图像
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:  # 捕获并打印任何异常
            print(repr(e))

    # 更新模型文件选项
    def update_model_items(self):
        selected_task = self.comboBox_task.currentText()
        self.comboBox_model.clear()

        if selected_task == "骨龄评估":
            self.comboBox_model.addItems(self.pt_list)
        elif selected_task == "检测":
            filtered_list = [file for file in self.pt_list if not file.endswith('-cls.pt')]
            self.comboBox_model.addItems(filtered_list)
        elif selected_task == "分类":
            filtered_list = [file for file in self.pt_list if file.endswith('-cls.pt')]
            self.comboBox_model.addItems(filtered_list)

    def search_pt(self):
        pt_list = os.listdir('./pt')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize('./pt/' + x))

        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.update_model_items()

    # 切换模型时更新 YOLOv8 线程的权重路径，并显示切换信息
    def change_model(self, x):
        self.model_type = self.comboBox_model.currentText()
        self.det_thread.weights = "./pt/%s" % self.model_type
        self.bottom_msg('Change model to %s' % x)

    # 读取配置文件，初始化界面参数
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
        self.det_thread.rate_check = check
        self.checkBox_autosave.setCheckState(savecheck)
        self.is_save()

    # 判断是否勾选保存结果的复选框，更新 YOLOv5 线程的保存路径
    def is_save(self):
        if self.checkBox_autosave.isChecked():
            self.det_thread.save_fold = './result'
        else:
            self.det_thread.save_fold = None

    # 判断是否勾选限制检测速率的复选框，更新 YOLOv5 线程的速率控制状态
    def checkrate(self):
        if self.checkBox_enable.isChecked():
            self.det_thread.rate_check = True
        else:
            self.det_thread.rate_check = False

    # 开始或继续检测时触发事件，更新 YOLOv5 线程的状态和界面显示
    def run_or_continue(self):
        self.det_thread.jump_out = False
        if self.runButton.isChecked():
            self.checkBox_autosave.setEnabled(False)
            self.det_thread.is_continue = True
            if not self.det_thread.isRunning():
                self.det_thread.start()
            source = os.path.basename(self.det_thread.source)
            source = 'camera' if source.isnumeric() else source
            self.bottom_msg('Detecting >> model：{}，file：{}'.
                            format(os.path.basename(self.det_thread.weights),
                                   source))
        else:
            self.det_thread.is_continue = False
            self.bottom_msg('Pause')

    # 停止检测时触发事件，更新 YOLOv5 线程的状态和界面显示
    def stop(self):
        self.det_thread.jump_out = True
        self.checkBox_autosave.setEnabled(True)

    # 在底部消息标签中显示统计信息
    def bottom_msg(self, msg):
        self.label_bottom.setText(msg)
        # self.qtimer.start(3000)

    # 在底部消息标签中显示消息，并根据消息设置界面状态
    def show_msg(self, msg):
        self.runButton.setChecked(Qt.Unchecked)
        self.bottom_msg(msg)
        if msg == "Finished":
            self.checkBox_autosave.setEnabled(True)

    # 在结果窗口中显示统计信息
    def show_statistic(self, statistic_dic):
        try:
            self.listWidget.clear()
            statistic_dic = sorted(statistic_dic.items(), key=lambda x: x[1], reverse=True)
            statistic_dic = [i for i in statistic_dic if i[1] > 0]
            results = [' ' + str(i[0]) + '：' + str(i[1]) for i in statistic_dic]
            self.listWidget.addItems(results)

        except Exception as e:
            print(repr(e))

    # 最大化或还原窗口时触发事件
    def max_or_restore(self):
        if self.maxButton.isChecked():
            self.window().showMaximized()
        else:
            self.window().showNormal()

    # 鼠标按下事件，用于移动窗口
    def mousePressEvent(self, event):
        self.m_Position = event.pos()
        if event.button() == Qt.LeftButton:
            if 0 < self.m_Position.x() < self.groupBox.pos().x() + self.groupBox.width() and \
                    0 < self.m_Position.y() < self.groupBox.pos().y() + self.groupBox.height():
                self.m_flag = True

    # 鼠标移动事件，用于移动窗口
    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)

    # 鼠标释放事件，用于移动窗口
    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False

    # 关闭窗口时触发事件，保存配置文件并显示提示信息
    def closeEvent(self, event):
        self.det_thread.jump_out = True
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
        TipsMessageBox(
            self.closeButton, title='Tips', text='Closing the program', time=2000, auto=True).exec_()
        sys.exit(0)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myForm = MainWindow()
    myForm.show()
    app.exec()
