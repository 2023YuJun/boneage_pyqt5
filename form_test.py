import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QHBoxLayout, QFileDialog, QComboBox, QStatusBar
from PyQt5.QtGui import QPixmap, QPainter, QColor
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

    def __init__(self, start_point, end_point, category_color):
        """
        初始化BoundingBox实例。

        参数:
            start_point (QPointF): 方框的起始点。
            end_point (QPointF): 方框的终止点。
            category_color (QColor): 方框的颜色。
        """
        self.start_point = QPointF(start_point)
        self.end_point = QPointF(end_point)
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
            self.draw_corners(painter, scale_factor, offset)

    def draw_corners(self, painter, scale_factor, offset):
        """
        绘制方框的四个角，用于调整大小。

        参数:
            painter (QPainter): 用于绘制的QPainter对象。
            scale_factor (QPointF): 图像的缩放比例。
            offset (QPointF): 图像的偏移量。
        """
        corner_size = 8
        corner_color = QColor(self.category_color.red(), self.category_color.green(), self.category_color.blue(), 192)  # 75%不透明度

        corners = self.get_corners(scale_factor, offset)

        painter.setBrush(corner_color)
        painter.setPen(Qt.NoPen)

        for corner_rect in corners.values():
            painter.drawRect(corner_rect)

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
            "top_left": QRectF(top_left + offset - QPointF(corner_size / 2, corner_size / 2), QSizeF(corner_size, corner_size)),
            "top_right": QRectF(top_right + offset - QPointF(corner_size / 2, corner_size / 2), QSizeF(corner_size, corner_size)),
            "bottom_left": QRectF(bottom_left + offset - QPointF(corner_size / 2, corner_size / 2), QSizeF(corner_size, corner_size)),
            "bottom_right": QRectF(bottom_right + offset - QPointF(corner_size / 2, corner_size / 2), QSizeF(corner_size, corner_size)),
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
        new_pos_scaled = QPointF((new_pos.x() - offset.x()) / scale_factor.x(), (new_pos.y() - offset.y()) / scale_factor.y())

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

        # 为 QLabel 添加黑色边框
        self.setStyleSheet("border: 2px solid black;")

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


class MainWindow(QMainWindow):
    """
    MainWindow类继自QMainWindow，是图像标注工具的主窗口。
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像标注工具")
        self.setGeometry(100, 100, 800, 600)

        # 初始化UI和状态栏
        self.init_ui()
        self.init_status_bar()

        # 当前选择的类别和颜色
        self.current_category_color = QColor(Qt.red)

        # 管理多个标注框
        self.current_box = None
        self.dragging_box = None
        self.dragging_offset = None

    def init_ui(self):
        """初始化主窗口UI控件"""
        # 创建图像显示区域
        self.image_label = ImageLabel(self)
        self.image_label.setFocusPolicy(Qt.StrongFocus)
        self.image_label.mousePressEvent = self.on_mouse_press
        self.image_label.mouseMoveEvent = self.on_mouse_move
        self.image_label.mouseReleaseEvent = self.on_mouse_release

        # 创建按钮和下拉框
        buttons = [
            ("打开图像", self.open_image),
            ("保存标注文件", self.save_annotations),
            ("保存裁剪图像", self.save_cropped_images),
            ("清除当前类别方框", self.clear_current_category_boxes)
        ]
        button_layout = self.create_button_layout(buttons)

        self.category_combo = QComboBox()
        for category, color in [("类别1", Qt.red), ("类别2", Qt.green), ("类别3", Qt.blue)]:
            self.category_combo.addItem(category, QColor(color))
        self.category_combo.currentIndexChanged.connect(self.change_category)
        button_layout.addWidget(self.category_combo)

        # 设置布局
        layout = QVBoxLayout()
        layout.addLayout(button_layout)
        layout.addWidget(self.image_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def create_button_layout(self, buttons):
        """创建包含按钮的水平布局"""
        layout = QHBoxLayout()
        for label, callback in buttons:
            button = QPushButton(label)
            button.clicked.connect(callback)
            layout.addWidget(button)
        return layout

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

    def open_image(self):
        """打开图像文件并加载到标签中。"""
        file_name, _ = QFileDialog.getOpenFileName(self, "打开图像文件", "", "Image Files (*.png *.jpg *.bmp)")
        if file_name:
            self.load_image(file_name)

    def load_image(self, image_path):
        """加载图像并在标签中显示。"""
        self.image_label.setPixmap(QPixmap(image_path))

    def save_annotations(self):
        """保存标注信息到文件。"""
        with open("annotations.txt", "w") as file:
            for box in BoundingBox.bounding_boxes:
                start_point, end_point = box.start_point, box.end_point
                file.write(
                    f"{int(start_point.x())},{int(start_point.y())},{int(end_point.x())},{int(end_point.y())},{box.category_color.name()}\n")

    def save_cropped_images(self):
        """保存每个类别的裁剪图像。"""
        for i, box in enumerate(BoundingBox.bounding_boxes):
            rect = QRectF(box.start_point, box.end_point).normalized()
            cropped_pixmap = self.image_label.pixmap.copy(rect.toRect())
            cropped_pixmap.save(f"cropped_image_{i}.png")

    def clear_current_category_boxes(self):
        """清除当前选择类别的所有方框。"""
        BoundingBox.bounding_boxes = [box for box in BoundingBox.bounding_boxes if
                                      box.category_color != self.current_category_color]
        self.image_label.update()

    def change_category(self):
        """更改当前选择的类别颜色。"""
        self.current_category_color = self.category_combo.currentData()

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
            self.current_box = BoundingBox(pos, pos, self.current_category_color)
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
