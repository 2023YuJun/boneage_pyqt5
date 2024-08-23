import os
import json
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QHBoxLayout, \
    QFileDialog, QComboBox, QStatusBar, QCheckBox, QMessageBox, QSpinBox, QTabWidget
from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont
from PyQt5.QtCore import Qt, QRectF, QPointF, QSizeF


class BoundingBox:
    """
    BoundingBox类用于表示图像上的标注框。

    类属性:
        bounding_boxes (list): 存储所有BoundingBox实例的列表。
        selected_box (BoundingBox): 当前选中的BoundingBox实例。

    实例属性:
        start_point (QPointF): 方框的起始点。
        end_point (QPointF): 方框的终止点。
        category_color (QColor): 方框的颜色，代表不同类别。
        selected (bool): 方框是否被选中。
        dragging_corner (str): 当前正在拖动的角落，用于调整方框大小。
    """

    bounding_boxes = []  # 存储所有BoundingBox实例
    selected_box = None  # 当前选中的BoundingBox实例

    def __init__(self, start_point, end_point, category_name, category_color):
        """
        初始化BoundingBox实例。

        参数:
            start_point (QPointF): 方框的起始点。
            end_point (QPointF): 方框的终止点。
            category_color (QColor): 方框的颜色。
        """
        self.start_point = QPointF(start_point)
        self.end_point = QPointF(end_point)
        self.category_name = category_name
        self.category_color = QColor(category_color)
        self.selected = False
        self.dragging_corner = None
        BoundingBox.bounding_boxes.append(self)  # 创建时自动添加到bounding_boxes列表中

    def draw(self, painter, scale_factor, offset):
        """
        绘制方框。

        参数:
            painter (QPainter): 用于绘制的QPainter对象。
            scale_factor (QPointF): 图像的缩放比例。
            offset (QPointF): 图像的偏移量。
        """
        scaled_start_point = QPointF(self.start_point.x() * scale_factor.x(), self.start_point.y() * scale_factor.y())
        scaled_end_point = QPointF(self.end_point.x() * scale_factor.x(), self.end_point.y() * scale_factor.y())
        rect = QRectF(scaled_start_point + offset, scaled_end_point + offset).normalized()

        brush_color = QColor(self.category_color.red(), self.category_color.green(), self.category_color.blue(),
                             128 if self.selected else 64)  # 50%透明度被选中，25%透明度未选中

        painter.setPen(self.category_color)
        painter.setBrush(brush_color)
        painter.drawRect(rect)

        if self.selected:
            self.draw_label(painter, rect)
            self.draw_corners(painter, scale_factor, offset)

    def draw_corners(self, painter, scale_factor, offset):
        """
        绘制方框的四个角，用于调整大小。

        参数:
            painter (QPainter): 用于绘制的QPainter对象。
            scale_factor (QPointF): 图像的缩放比例。
            offset (QPointF): 图像的偏移量。
        """
        corner_color = QColor(self.category_color.red(), self.category_color.green(), self.category_color.blue(),
                              192)  # 75%不透明度

        corners = self.get_corners(scale_factor, offset)

        painter.setBrush(corner_color)
        painter.setPen(Qt.NoPen)

        for corner_rect in corners.values():
            painter.drawRect(corner_rect)

    def draw_label(self, painter, rect):
        """在方框左上角显示类别名"""
        corner_color = QColor(self.category_color.red(), self.category_color.green(), self.category_color.blue(),
                              192)
        font = QFont()
        font.setPointSize(12)
        painter.setFont(font)
        painter.setPen(corner_color)
        painter.drawText(rect.topLeft() + QPointF(5, -8), self.category_name)

    def get_corners(self, scale_factor, offset):
        """
        获取方框四个角的矩形区域。

        参数:
            scale_factor (QPointF): 图像的缩放比例。
            offset (QPointF): 图像的偏移量。

        返回:
            dict: 包含四个角区域的字典，键为角的名称，值为QRectF对象。
        """
        corner_size = 8
        top_left = QPointF(self.start_point.x() * scale_factor.x(), self.start_point.y() * scale_factor.y())
        top_right = QPointF(self.end_point.x() * scale_factor.x(), self.start_point.y() * scale_factor.y())
        bottom_left = QPointF(self.start_point.x() * scale_factor.x(), self.end_point.y() * scale_factor.y())
        bottom_right = QPointF(self.end_point.x() * scale_factor.x(), self.end_point.y() * scale_factor.y())

        corners = {
            "top_left": QRectF(top_left + offset - QPointF(corner_size / 2, corner_size / 2),
                               QSizeF(corner_size, corner_size)),
            "top_right": QRectF(top_right + offset - QPointF(corner_size / 2, corner_size / 2),
                                QSizeF(corner_size, corner_size)),
            "bottom_left": QRectF(bottom_left + offset - QPointF(corner_size / 2, corner_size / 2),
                                  QSizeF(corner_size, corner_size)),
            "bottom_right": QRectF(bottom_right + offset - QPointF(corner_size / 2, corner_size / 2),
                                   QSizeF(corner_size, corner_size)),
        }

        return corners

    def get_corner(self, point, scale_factor, offset):
        """
        检查鼠标点击的位置是否在方框的四个角上。

        参数:
            point (QPointF): 鼠标点击的位置。
            scale_factor (QPointF): 图像的缩放比例。
            offset (QPointF): 图像的偏移量。

        返回:
            str: 鼠标点击的角的名称，如果不在角上则返回None。
        """
        corners = self.get_corners(scale_factor, offset)

        for corner, area in corners.items():
            if area.contains(point):
                return corner
        return None

    def contains(self, point, scale_factor, offset):
        """
        检查点是否在方框内。

        参数:
            point (QPointF): 需要检查的点。
            scale_factor (QPointF): 图像的缩放比例。
            offset (QPointF): 图像的偏移量。

        返回:
            bool: 如果点在方框内，返回True；否则返回False。
        """
        scaled_start_point = QPointF(self.start_point.x() * scale_factor.x(), self.start_point.y() * scale_factor.y())
        scaled_end_point = QPointF(self.end_point.x() * scale_factor.x(), self.end_point.y() * scale_factor.y())
        rect = QRectF(scaled_start_point + offset, scaled_end_point + offset).normalized()
        return rect.contains(point)

    def move(self, offset):
        """
        移动方框。

        参数:
            offset (QPointF): 移动的偏移量。
        """
        self.start_point += offset
        self.end_point += offset

    def resize(self, corner, new_pos, scale_factor, offset):
        """
        调整方框大小。

        参数:
            corner (str): 被拖动的角的名称。
            new_pos (QPointF): 新的角的位置。
            scale_factor (QPointF): 图像的缩放比例。
            offset (QPointF): 图像的偏移量。
        """
        new_pos_scaled = QPointF((new_pos.x() - offset.x()) / scale_factor.x(),
                                 (new_pos.y() - offset.y()) / scale_factor.y())

        if corner == "top_left":
            self.start_point = new_pos_scaled
        elif corner == "bottom_right":
            self.end_point = new_pos_scaled
        elif corner == "top_right":
            self.start_point.setY(new_pos_scaled.y())
            self.end_point.setX(new_pos_scaled.x())
        elif corner == "bottom_left":
            self.start_point.setX(new_pos_scaled.x())
            self.end_point.setY(new_pos_scaled.y())

    def width(self):
        """
        获取方框的宽度。

        返回:
            float: 方框的宽度。
        """
        return abs(self.end_point.x() - self.start_point.x())

    def height(self):
        """
        获取方框的高度。

        返回:
            float: 方框的高度。
        """
        return abs(self.end_point.y() - self.start_point.y())

    def to_image_coordinates(self, scale_factor):
        """
        将方框的坐标转换为图像坐标。

        参数:
            scale_factor (QPointF): 图像的缩放比例。

        返回:
            tuple: 起始点和终止点的图像坐标，形式为(QPointF, QPointF)。
        """
        start_point = QPointF(self.start_point.x() / scale_factor.x(), self.start_point.y() / scale_factor.y())
        end_point = QPointF(self.end_point.x() / scale_factor.x(), self.end_point.y() / scale_factor.y())
        return start_point, end_point

    @classmethod
    def deselect_all(cls):
        """取消选择所有BoundingBox实例"""
        for box in cls.bounding_boxes:
            box.selected = False
        cls.selected_box = None

    @classmethod
    def remove_selected(cls):
        """删除当前选中的BoundingBox实例"""
        if cls.selected_box:
            cls.bounding_boxes.remove(cls.selected_box)
            cls.selected_box = None


class ImageLabel(QLabel):
    """
    ImageLabel类继承自QLabel，用于显示图像并管理图像上的标注框。

    属性:
        pixmap (QPixmap): 要显示的图像。
        image_offset (QPointF): 图像在QLabel中的偏移量。
        image_scaled_size (QSizeF): 缩放后的图像大小。
    """

    def __init__(self, parent=None):
        """
        初始化ImageLabel实例。

        参数:
            parent (QWidget): 父组件。
        """
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(False)
        self.pixmap = None
        self.image_offset = QPointF(0, 0)  # 图像在label中的偏移量
        self.image_scaled_size = QSizeF(0, 0)  # 缩放后的图像大小

    def setPixmap(self, pixmap):
        """
        设置要显示的图像。

        参数:
            pixmap (QPixmap): 要显示的图像。
        """
        self.pixmap = pixmap
        self.update()

    def paintEvent(self, event):
        """
        重写QLabel的paintEvent方法，用于绘制图像和标注框。

        参数:
            event (QPaintEvent): 绘制事件。
        """
        super().paintEvent(event)
        if self.pixmap:
            # 计算图像在 QLabel 中的缩放比例和偏移量
            self.image_scaled_size = self.pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation).size()
            self.image_offset = QPointF(
                (self.width() - self.image_scaled_size.width()) // 2,
                (self.height() - self.image_scaled_size.height()) // 2
            )
            painter = QPainter(self)
            painter.drawPixmap(self.image_offset.toPoint(), self.pixmap.scaled(self.image_scaled_size))
            scale_factor = QPointF(self.image_scaled_size.width() / self.pixmap.width(),
                                   self.image_scaled_size.height() / self.pixmap.height())
            # 绘制所有标注框，并考虑图像的缩放比例和偏移量
            for box in BoundingBox.bounding_boxes:
                box.draw(painter, scale_factor, self.image_offset)

    def to_image_coordinates(self, pos):
        """
        将窗口坐标转换为相对于缩放后图像的坐标。

        参数:
            pos (QPointF): 窗口坐标。

        返回:
            QPointF: 相对于图像的坐标。
        """
        image_x = (pos.x() - self.image_offset.x()) / (self.image_scaled_size.width() / self.pixmap.width())
        image_y = (pos.y() - self.image_offset.y()) / (self.image_scaled_size.height() / self.pixmap.height())
        return QPointF(image_x, image_y)

    def is_inside_image(self, pos):
        """
        检查点是否在图像内。

        参数:
            pos (QPointF): 要检查的点。

        返回:
            bool: 如果点在图像内，返回True；否则返回False。
        """
        return self.image_offset.x() <= pos.x() < self.image_offset.x() + self.image_scaled_size.width() and \
            self.image_offset.y() <= pos.y() < self.image_offset.y() + self.image_scaled_size.height()


def convert_to_imagelabel(qlabel):
    """ 将Qlabel对象转换为ImageLabel对象 """
    # 获取QLabel的父对象
    parent = qlabel.parent()

    # 获取QLabel的原始属性
    original_geometry = qlabel.geometry()  # 保存位置和大小
    original_style = qlabel.styleSheet()  # 保存样式
    original_layout = qlabel.parent().layout()  # 获取QLabel的父布局
    original_index = original_layout.indexOf(qlabel)  # 获取QLabel在布局中的索引

    # 从布局中移除原QLabel
    original_layout.removeWidget(qlabel)
    qlabel.deleteLater()  # 删除原来的QLabel

    # 创建新的ImageLabel
    new_imagelabel = ImageLabel(parent)

    # 恢复原QLabel的属性到新的ImageLabel
    new_imagelabel.setGeometry(original_geometry)
    new_imagelabel.setStyleSheet(original_style)

    # 将新的ImageLabel添加回布局中
    original_layout.insertWidget(original_index, new_imagelabel)

    return new_imagelabel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像标注工具")
        self.setGeometry(100, 100, 1000, 600)  # 增加窗口初始宽度

        self.config_file = "config.json"
        self.save_path = self.load_save_path()

        # 初始化UI和状态栏
        self.init_ui()
        self.init_status_bar()
        self.image_label = convert_to_imagelabel(self.image_label)
        self.image_label.setFocusPolicy(Qt.StrongFocus)
        self.image_label.mousePressEvent = self.on_mouse_press
        self.image_label.mouseMoveEvent = self.on_mouse_move
        self.image_label.mouseReleaseEvent = self.on_mouse_release

        # 当前选择的类别和颜色，初始设为第一个类别
        self.current_category_color = QColor(Qt.red)
        self.current_category_name = "类别1"

        # 管理多个标注框
        self.current_box = None
        self.dragging_box = None
        self.dragging_offset = None

    def load_save_path(self):
        """加载保存路径，如果存在则返回路径，否则返回None"""
        if os.path.exists(self.config_file):
            with open(self.config_file, "r") as f:
                config = json.load(f)
                return config.get("save_path", None)
        return None

    def save_save_path(self, path):
        """保存路径到config.json文件"""
        with open(self.config_file, "w") as f:
            json.dump({"save_path": path}, f)

    def init_ui(self):
        """初始化主窗口UI控件"""
        # 创建图像显示区域
        self.image_label = QLabel(self)

        # 创建TabWidget
        self.tab_widget = QTabWidget()
        self.tab_widget.setFixedWidth(200)  # 设置TabWidget的固定宽度
        self.tab_widget.addTab(self.create_first_tab(), "9个类别")
        self.tab_widget.addTab(self.create_second_tab(), "7个类别")

        # 连接Tab切换信号到槽函数
        self.tab_widget.currentChanged.connect(self.on_tab_change)

        # 创建左边按钮布局
        buttons = [
            ("打开图像", self.open_image),
            ("清除当前类别方框", self.clear_current_category_boxes)
        ]
        button_layout = self.create_button_layout(buttons)

        # 设置主布局为水平布局，将TabWidget放在左边，ImageLabel放在右边
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        left_layout.addLayout(button_layout)
        left_layout.addWidget(self.tab_widget)

        main_layout.addLayout(left_layout)
        main_layout.addWidget(self.image_label)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def create_button_layout(self, buttons):
        """创建包含按钮的垂直布局"""
        layout = QVBoxLayout()  # 将按钮竖直排列
        for label, callback in buttons:
            button = QPushButton(label)
            button.clicked.connect(callback)
            layout.addWidget(button)
        return layout

    def create_first_tab(self):
        """创建包含9个类别checkbox和SpinBox的选项卡"""
        widget = QWidget()
        layout = QVBoxLayout()

        self.colors_9 = [Qt.red, Qt.green, Qt.blue, Qt.yellow, Qt.cyan, Qt.magenta, Qt.darkRed, Qt.darkGreen,
                         Qt.darkBlue]
        self.category_names_9 = ["类别1", "类别2", "类别3", "类别4", "类别5", "类别6", "类别7", "类别8", "类别9"]
        self.checkboxes_9 = []

        for i, (color, name) in enumerate(zip(self.colors_9, self.category_names_9), 1):
            h_layout = QHBoxLayout()
            checkbox = QCheckBox(name)
            checkbox.setStyleSheet(f"color: {QColor(color).name()};")
            checkbox.clicked.connect(self.on_checkbox_checked)
            spinbox = QSpinBox()
            spinbox.setMinimum(1)
            spinbox.setMaximum(100)
            h_layout.addWidget(checkbox)
            h_layout.addWidget(spinbox)
            layout.addLayout(h_layout)
            self.checkboxes_9.append((checkbox, spinbox, color, name))

        save_button = QPushButton("保存裁剪图像")
        save_button.clicked.connect(self.save_cropped_images)
        layout.addWidget(save_button)

        clear_boxes_button = QPushButton("清除所有方框")
        clear_boxes_button.clicked.connect(self.clear_all_boxes)
        layout.addWidget(clear_boxes_button)

        # 默认选中第一个类别
        self.checkboxes_9[0][0].setChecked(True)

        widget.setLayout(layout)
        return widget

    def create_second_tab(self):
        """创建包含7个类别checkbox的选项卡"""
        widget = QWidget()
        layout = QVBoxLayout()

        self.colors_7 = self.colors_9[:7]
        self.category_names_7 = ["类别1", "类别2", "类别3", "类别4", "类别5", "类别6", "类别7"]
        self.checkboxes_7 = []

        for i, color in enumerate(self.colors_7, 1):
            checkbox = QCheckBox(f"类别{i}")
            checkbox.setStyleSheet(f"color: {QColor(color).name()};")
            checkbox.clicked.connect(self.on_checkbox_checked)
            layout.addWidget(checkbox)
            self.checkboxes_7.append((checkbox, color, f"类别{i}"))

        save_button = QPushButton("保存注释文件")
        save_button.clicked.connect(self.save_annotations)
        layout.addWidget(save_button)

        clear_boxes_button = QPushButton("清除所有方框")
        clear_boxes_button.clicked.connect(self.clear_all_boxes)
        layout.addWidget(clear_boxes_button)

        # 默认选中第一个类别
        self.checkboxes_7[0][0].setChecked(True)

        widget.setLayout(layout)
        return widget

    def on_tab_change(self):
        """处理选项卡切换事件"""
        reply = QMessageBox.question(
            self,
            "保存工作",
            "你想在切换选项卡前保存注释文件和裁剪图像吗?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.save_annotations()
            self.save_cropped_images()

        # 清除所有方框
        BoundingBox.bounding_boxes.clear()
        self.image_label.update()

    def on_checkbox_checked(self):
        """处理QCheckBox被点击时的逻辑，确保只有一个类别被选中"""
        if self.tab_widget.currentIndex() == 0:  # 9个类别的选项卡
            for checkbox, spinbox, color, name in self.checkboxes_9:
                if checkbox.isChecked() and checkbox != self.sender():
                    checkbox.setChecked(False)
                elif checkbox.isChecked() and checkbox == self.sender():
                    self.current_category_color = QColor(color)  # 设置当前类别颜色
                    self.current_category_name = name  # 设置当前类别名称
        else:  # 7个类别的选项卡
            for checkbox, color, name in self.checkboxes_7:
                if checkbox.isChecked() and checkbox != self.sender():
                    checkbox.setChecked(False)
                elif checkbox.isChecked() and checkbox == self.sender():
                    self.current_category_color = QColor(color)  # 设置当前类别颜色
                    self.current_category_name = name  # 设置当前类别名称

    def open_image(self):
        """打开图像文件并加载到标签中。"""
        file_name, _ = QFileDialog.getOpenFileName(self, "打开图像文件", "", "Image Files (*.png *.jpg *.bmp)")
        if file_name:
            self.image_label.setPixmap(QPixmap(file_name))
            self.current_image_path = file_name

    def save_annotations(self):
        """保存标注信息到文件。"""
        if self.save_path:
            selected_path = QFileDialog.getExistingDirectory(self, "选择保存路径", self.save_path)
        else:
            selected_path = QFileDialog.getExistingDirectory(self, "选择保存路径")

        if selected_path:
            self.save_path = selected_path
            self.save_save_path(self.save_path)
        else:
            return

        # 检查detect_datasets目录
        detect_dir = os.path.join(self.save_path, "detect_datasets")
        os.makedirs(detect_dir, exist_ok=True)

        # 检查images和labels目录
        images_dir = os.path.join(detect_dir, "images")
        labels_dir = os.path.join(detect_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        # 保存图像到images目录
        image_name = os.path.basename(self.current_image_path)
        image_save_path = os.path.join(images_dir, image_name)
        self.image_label.pixmap.save(image_save_path)

        # 保存注释到labels目录
        label_save_path = os.path.join(labels_dir, os.path.splitext(image_name)[0] + ".txt")
        scale_factor = QPointF(self.image_label.image_scaled_size.width() / self.image_label.pixmap.width(),
                               self.image_label.image_scaled_size.height() / self.image_label.pixmap.height())

        with open(label_save_path, "w") as file:
            for box in BoundingBox.bounding_boxes:
                start_point, end_point = box.to_image_coordinates(scale_factor)
                x_center = (start_point.x() + end_point.x()) / 2 / self.image_label.pixmap.width()
                y_center = (start_point.y() + end_point.y()) / 2 / self.image_label.pixmap.height()
                width = abs(end_point.x() - start_point.x()) / self.image_label.pixmap.width()
                height = abs(end_point.y() - start_point.y()) / self.image_label.pixmap.height()
                category_index = self.get_category_names().index(box.category_name)
                file.write(f"{category_index} {x_center} {y_center} {width} {height}\n")

        # 生成data.yaml
        data_yaml_path = os.path.join(detect_dir, "data.yaml")
        with open(data_yaml_path, "w", encoding='utf-8') as yaml_file:
            category_names = self.get_category_names()
            yaml_file.write(f"nc: {len(category_names)}\n")
            yaml_file.write(f"names: {category_names}\n")

    def get_category_names(self):
        """获取当前选项卡的类别名列表"""
        if self.tab_widget.currentIndex() == 0:
            return self.category_names_9
        else:
            return self.category_names_7

    def save_cropped_images(self):
        """保存每个类别的裁剪图像。"""
        if self.save_path:
            selected_path = QFileDialog.getExistingDirectory(self, "选择保存路径", self.save_path)
        else:
            selected_path = QFileDialog.getExistingDirectory(self, "选择保存路径")

        if selected_path:
            self.save_path = selected_path
            self.save_save_path(self.save_path)
        else:
            return

        # 检查classify_datasets目录
        classify_dir = os.path.join(self.save_path, "classify_datasets")
        os.makedirs(classify_dir, exist_ok=True)

        # 遍历所有BoundingBox，按类别和等级保存裁剪图像
        for box in BoundingBox.bounding_boxes:
            # 遍历所有9个类别的checkbox和spinbox
            for checkbox, spinbox, color, name in self.checkboxes_9:
                if box.category_name == name:
                    category_dir = os.path.join(classify_dir, name)
                    os.makedirs(category_dir, exist_ok=True)

                    # 检查SpinBox值目录
                    value_dir = os.path.join(category_dir, str(spinbox.value()))
                    os.makedirs(value_dir, exist_ok=True)

                    # 保存裁剪图像
                    rect = QRectF(box.start_point, box.end_point).normalized()
                    cropped_pixmap = self.image_label.pixmap.copy(rect.toRect())

                    # 确保文件名唯一
                    image_count = len(os.listdir(value_dir))
                    cropped_image_path = os.path.join(value_dir, f"{image_count + 1}.png")
                    cropped_pixmap.save(cropped_image_path)

    def clear_current_category_boxes(self):
        """清除当前选择类别的所有方框。"""
        BoundingBox.bounding_boxes = [box for box in BoundingBox.bounding_boxes if
                                      box.category_color != self.current_category_color]
        self.image_label.update()

    def clear_all_boxes(self):
        """清除所有方框。"""
        BoundingBox.bounding_boxes.clear()
        self.image_label.update()

    def init_status_bar(self):
        """初始化状态栏"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.mouse_position_label = QLabel("鼠标位置: ")
        self.start_position_label = QLabel("方框起始位置: ")
        self.end_position_label = QLabel("方框终止位置: ")
        self.box_size_label = QLabel("方框大小: ")
        for widget in [self.mouse_position_label, self.start_position_label, self.end_position_label,
                       self.box_size_label]:
            self.status_bar.addWidget(widget)

    def on_mouse_press(self, event):
        """鼠标按下事件处理。"""
        if not self.image_label.is_inside_image(event.pos()):
            return

        pos = self.image_label.to_image_coordinates(event.pos())
        scale_factor = QPointF(self.image_label.image_scaled_size.width() / self.image_label.pixmap.width(),
                               self.image_label.image_scaled_size.height() / self.image_label.pixmap.height())

        if event.button() == Qt.LeftButton:
            for box in BoundingBox.bounding_boxes:
                if box.selected:
                    corner = box.get_corner(event.pos(), scale_factor, self.image_label.image_offset)
                    if corner:
                        box.dragging_corner = corner
                        self.dragging_box = box
                        return

            for box in BoundingBox.bounding_boxes:
                if box.contains(event.pos(), scale_factor, self.image_label.image_offset):
                    BoundingBox.deselect_all()
                    self.dragging_box = box
                    box.selected = True
                    self.dragging_offset = pos - box.start_point
                    BoundingBox.selected_box = box
                    self.update_status_bar(box, event.pos())
                    self.image_label.update()
                    return

            BoundingBox.deselect_all()
            self.current_box = BoundingBox(pos, pos, self.current_category_name, self.current_category_color)
            self.update_status_bar(self.current_box, event.pos())

    def on_mouse_move(self, event):
        """鼠标移动事件处理。"""
        if not self.image_label.is_inside_image(event.pos()):
            return

        pos = self.image_label.to_image_coordinates(event.pos())

        if self.current_box:
            self.current_box.end_point = pos
            self.update_status_bar(self.current_box, event.pos())
            self.image_label.update()
        elif self.dragging_box:
            scale_factor = QPointF(self.image_label.image_scaled_size.width() / self.image_label.pixmap.width(),
                                   self.image_label.image_scaled_size.height() / self.image_label.pixmap.height())
            if self.dragging_box.dragging_corner:
                self.dragging_box.resize(self.dragging_box.dragging_corner, event.pos(), scale_factor,
                                         self.image_label.image_offset)
            else:
                offset = pos - (self.dragging_box.start_point + self.dragging_offset)
                self.dragging_box.move(offset)
                self.dragging_offset = pos - self.dragging_box.start_point
            self.update_status_bar(self.dragging_box, event.pos())
            self.image_label.update()

    def on_mouse_release(self, event):
        """鼠标释放事件处理。"""
        if self.current_box:
            if self.current_box.width() < 2 or self.current_box.height() < 2:
                BoundingBox.bounding_boxes.remove(self.current_box)
            self.current_box = None
        elif self.dragging_box:
            self.dragging_box.dragging_corner = None
            self.dragging_box = None
            self.dragging_offset = None
        self.image_label.update()

    def keyPressEvent(self, event):
        """键盘按下事件处理。"""
        if event.key() in [Qt.Key_Delete, Qt.Key_Backspace]:
            BoundingBox.remove_selected()
            self.image_label.update()

    def update_status_bar(self, box, mouse_pos=None):
        """更新状态栏显示的方框信息及鼠标位置。"""
        start_pos = box.start_point
        end_pos = box.end_point

        # 将坐标值精确到两位小数
        start_x = f"{start_pos.x():.2f}"
        start_y = f"{start_pos.y():.2f}"
        end_x = f"{end_pos.x():.2f}"
        end_y = f"{end_pos.y():.2f}"

        self.start_position_label.setText(f"方框起始位置: ({start_x}, {start_y})")
        self.end_position_label.setText(f"方框终止位置: ({end_x}, {end_y})")
        self.box_size_label.setText(f"方框大小: {box.width():.2f} x {box.height():.2f}")

        if mouse_pos:
            image_coords = self.image_label.to_image_coordinates(mouse_pos)
            mouse_x = f"{image_coords.x():.2f}"
            mouse_y = f"{image_coords.y():.2f}"
            self.mouse_position_label.setText(f"鼠标位置: ({mouse_x}, {mouse_y})")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
