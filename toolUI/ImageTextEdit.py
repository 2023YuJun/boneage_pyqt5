import os
import requests
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QTextImageFormat, QTextCursor
from PyQt5.QtWidgets import QTextEdit, QVBoxLayout


class ImageTextEdit(QTextEdit):
    """
    ImageTextEdit 是一个自定义的 QTextEdit 类，支持拖放图片，并能够从剪贴板中插入图片。
    还提供了从 URL 下载图片的功能，并自动将其插入文本中。
    """

    def __init__(self, parent=None):
        """
        初始化 ImageTextEdit 类，设置接受拖放，并创建临时文件目录。

        :param parent: 父级窗口，默认为 None。
        """
        super(ImageTextEdit, self).__init__(parent)
        self.setAcceptDrops(True)
        self.temp_file_directory = 'temp_file/temp_image'
        # 确保临时文件目录存在
        os.makedirs(self.temp_file_directory, exist_ok=True)

    def insertFromMimeData(self, source):
        """
        从剪贴板或拖放的 MIME 数据中插入内容。

        :param source: QMimeData 对象，包含拖放或粘贴的数据。
        """
        if source.hasImage():
            # 如果数据中包含图像，则插入图像
            image = source.imageData()
            path = self.get_image_path_from_mime_data(source)
            self.insert_image(image, path)
        elif source.hasUrls():
            # 如果数据中包含 URL，则尝试从 URL 中插入图像
            for url in source.urls():
                self.insert_image_from_url(url)
        else:
            super().insertFromMimeData(source)

    def get_image_path_from_mime_data(self, source):
        """
        从 MIME 数据生成图像路径并保存图像。

        :param source: QMimeData 对象，包含拖放或粘贴的数据。
        :return: 保存的图像路径。
        """
        temp_path = self.generate_temp_image_path()
        image = source.imageData()
        image.save(temp_path)
        return temp_path

    def generate_temp_image_path(self):
        """
        生成一个唯一的临时图像路径。

        :return: 临时图像的路径。
        """
        count = 0
        while True:
            count += 1
            temp_filename = f'temp_image{count}.png'
            temp_path = os.path.join(self.temp_file_directory, temp_filename)
            if not os.path.exists(temp_path):
                return temp_path

    def insert_image(self, image, path):
        """
        插入图像到 QTextEdit 中。

        :param image: 要插入的 QImage 对象。
        :param path: 图像的路径，用于 QTextImageFormat。
        """
        if image.width() > self.width():
            # 如果图像宽度超过 QTextEdit 宽度，则缩放图像
            image = image.scaledToWidth(self.width())
        cursor = self.textCursor()
        image_format = QTextImageFormat()
        image_format.setName(path if path else "")
        image_format.setWidth(image.width())
        image_format.setHeight(image.height())
        cursor.insertImage(image_format)
        # 使用定时器替换 QTextEdit 为 ImageTextEdit
        QTimer.singleShot(100, lambda: replace_textedit_with_imagetextedit(self))

    def insert_image_from_url(self, url):
        """
        从 URL 插入图像到 QTextEdit 中。

        :param url: 包含图像的 URL。
        """
        if url.isLocalFile():
            # 如果是本地文件，则直接插入图像
            image = QImage(url.toLocalFile())
            self.insert_image(image, url.toLocalFile())
        else:
            # 如果是网络文件，则下载并插入图像
            response = requests.get(url.toString())
            if response.status_code == 200:
                image_data = response.content
                image = QImage()
                image.loadFromData(image_data)
                temp_path = self.generate_temp_image_path()
                image.save(temp_path)
                self.insert_image(image, temp_path)
            else:
                print(f"从 URL 加载图像失败: {url.toString()}")

    def cleanup_temp_images(self):
        """
        清理不再使用的临时图像文件。
        """
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.Start)
        image_paths = set()

        while True:
            cursor.select(QTextCursor.BlockUnderCursor)
            char_format = cursor.charFormat()
            if char_format.isImageFormat():
                image_format = char_format.toImageFormat()
                image_paths.add(image_format.name())

            cursor.movePosition(QTextCursor.NextBlock)

            if cursor.atEnd():
                break

        temp_files = [f for f in os.listdir(self.temp_file_directory) if f.endswith('.png')]

        for filename in temp_files:
            temp_path = os.path.join(self.temp_file_directory, filename)
            if temp_path not in image_paths:
                os.remove(temp_path)

    def dragEnterEvent(self, event):
        """
        处理拖动进入事件。

        :param event: 拖动事件对象。
        """
        if event.mimeData().hasImage() or event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event):
        """
        处理拖放事件。

        :param event: 拖放事件对象。
        """
        if event.mimeData().hasImage():
            self.insert_image(event.mimeData().imageData(), "")
            event.acceptProposedAction()
        elif event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                self.insert_image_from_url(url)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


def replace_textedit_with_imagetextedit(textedit):
    """
    替换指定的 QTextEdit 为 ImageTextEdit。

    :param textedit: 要替换的 QTextEdit 实例。
    :return: 新创建的 ImageTextEdit 实例。
    """
    parent_widget = textedit.parentWidget()
    if not parent_widget:
        raise ValueError("文本编辑控件没有父控件。")

    # 创建一个新的 ImageTextEdit 实例
    image_textedit = ImageTextEdit(parent_widget)

    # 复制现有 QTextEdit 的属性
    image_textedit.setTextCursor(textedit.textCursor())
    image_textedit.setTextInteractionFlags(textedit.textInteractionFlags())
    image_textedit.setGeometry(textedit.geometry())
    image_textedit.setObjectName(textedit.objectName())
    image_textedit.setStyleSheet(textedit.styleSheet())
    image_textedit.setFont(textedit.font())
    image_textedit.setSizePolicy(textedit.sizePolicy())
    image_textedit.setMinimumSize(textedit.minimumSize())
    image_textedit.setMaximumSize(textedit.maximumSize())
    image_textedit.setHtml(textedit.toHtml())
    image_textedit.setPlaceholderText(textedit.placeholderText())
    image_textedit.setAcceptRichText(textedit.acceptRichText())
    image_textedit.setVerticalScrollBarPolicy(textedit.verticalScrollBarPolicy())
    image_textedit.setHorizontalScrollBarPolicy(textedit.horizontalScrollBarPolicy())

    # 使用新的 ImageTextEdit 替换旧的 QTextEdit
    layout = parent_widget.layout()
    if layout:
        index = layout.indexOf(textedit)
        if index != -1:
            layout.replaceWidget(textedit, image_textedit)
            layout.removeWidget(textedit)
        else:
            raise ValueError("在布局中找不到文本编辑控件。")
    else:
        parent_widget.setLayout(QVBoxLayout())
        parent_widget.layout().addWidget(image_textedit)

    from mainwindow import MainWindow
    main_window = parent_widget.parent()
    while main_window.parent() is not None:
        main_window = main_window.parent()
    if isinstance(main_window, MainWindow):
        main_window.texteditUpdated.emit(image_textedit)

    return image_textedit
