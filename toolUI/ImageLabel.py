from PyQt5.QtCore import Qt, QRectF, QPointF, QSizeF
from PyQt5.QtGui import QPainter, QColor, QFont
from PyQt5.QtWidgets import QLabel


class BoundingBox:
    """
    BoundingBox 类用于表示图像上的标注框。

    类属性:
        bounding_boxes (list): 存储所有 BoundingBox 实例的列表。
        selected_box (BoundingBox): 当前选中的 BoundingBox 实例。

    实例属性:
        start_point (QPointF): 方框的起始点。
        end_point (QPointF): 方框的终止点。
        category_name (str): 方框的类别名称。
        category_color (QColor): 方框的颜色，表示不同类别。
        selected (bool): 方框是否被选中。
        dragging_corner (str): 当前正在拖动的角落，用于调整方框大小。
    """

    bounding_boxes = []  # 存储所有 BoundingBox 实例
    selected_box = None  # 当前选中的 BoundingBox 实例

    def __init__(self, start_point, end_point, category_name, category_color):
        """
        初始化 BoundingBox 实例。

        参数:
            start_point (QPointF): 方框的起始点。
            end_point (QPointF): 方框的终止点。
            category_name (str): 方框的类别名称。
            category_color (QColor): 方框的颜色。
        """
        self.start_point = QPointF(start_point)
        self.end_point = QPointF(end_point)
        self.category_name = category_name
        self.category_color = QColor(category_color)
        self.selected = False
        self.dragging_corner = None
        BoundingBox.bounding_boxes.append(self)  # 创建时自动添加到 bounding_boxes 列表中

    def draw(self, painter, scale_factor, offset):
        """
        绘制方框。

        参数:
            painter (QPainter): 用于绘制的 QPainter 对象。
            scale_factor (QPointF): 图像的缩放比例。
            offset (QPointF): 图像的偏移量。
        """
        scaled_start_point = QPointF(self.start_point.x() * scale_factor.x(), self.start_point.y() * scale_factor.y())
        scaled_end_point = QPointF(self.end_point.x() * scale_factor.x(), self.end_point.y() * scale_factor.y())
        rect = QRectF(scaled_start_point + offset, scaled_end_point + offset).normalized()

        brush_color = QColor(self.category_color.red(), self.category_color.green(), self.category_color.blue(),
                             128 if self.selected else 64)  # 被选中时透明度为50%，未选中时透明度为25%

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
            painter (QPainter): 用于绘制的 QPainter 对象。
            scale_factor (QPointF): 图像的缩放比例。
            offset (QPointF): 图像的偏移量。
        """
        corner_color = QColor(self.category_color.red(), self.category_color.green(), self.category_color.blue(),
                              192)  # 75% 不透明度

        corners = self.get_corners(scale_factor, offset)

        painter.setBrush(corner_color)
        painter.setPen(Qt.NoPen)

        for corner_rect in corners.values():
            painter.drawRect(corner_rect)

    def draw_label(self, painter, rect):
        """在方框的左上角显示类别名称"""
        corner_color = QColor(self.category_color.red(), self.category_color.green(), self.category_color.blue(),
                              192)
        font = QFont()
        font.setPointSize(12)
        painter.setFont(font)
        painter.setPen(corner_color)
        painter.drawText(rect.topLeft() + QPointF(0, -8), self.category_name)

    def get_corners(self, scale_factor, offset):
        """
        获取方框四个角的矩形区域。

        参数:
            scale_factor (QPointF): 图像的缩放比例。
            offset (QPointF): 图像的偏移量。

        返回:
            dict: 包含四个角区域的字典，键为角的名称，值为 QRectF 对象。
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
            str: 鼠标点击的角的名称，如果不在角上则返回 None。
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
            bool: 如果点在方框内，返回 True；否则返回 False。
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
            tuple: 起始点和终止点的图像坐标，形式为 (QPointF, QPointF)。
        """
        start_point = QPointF(self.start_point.x() / scale_factor.x(), self.start_point.y() / scale_factor.y())
        end_point = QPointF(self.end_point.x() / scale_factor.x(), self.end_point.y() / scale_factor.y())
        return start_point, end_point

    @classmethod
    def deselect_all(cls):
        """取消选择所有 BoundingBox 实例"""
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
