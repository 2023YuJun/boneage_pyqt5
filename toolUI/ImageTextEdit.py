import os

import requests
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QTextImageFormat, QTextCursor
from PyQt5.QtWidgets import QTextEdit, QVBoxLayout


class ImageTextEdit(QTextEdit):
    def __init__(self, parent=None):
        super(ImageTextEdit, self).__init__(parent)
        self.setAcceptDrops(True)
        self.temp_file_directory = 'temp_file/temp_image'
        # 确保临时文件目录存在
        os.makedirs(self.temp_file_directory, exist_ok=True)

    def insertFromMimeData(self, source):
        if source.hasImage():
            image = source.imageData()
            path = self.get_image_path_from_mime_data(source)
            self.insert_image(image, path)
        elif source.hasUrls():
            for url in source.urls():
                self.insert_image_from_url(url)
        else:
            super().insertFromMimeData(source)

    def get_image_path_from_mime_data(self, source):
        temp_path = self.generate_temp_image_path()
        image = source.imageData()
        image.save(temp_path)
        return temp_path

    def generate_temp_image_path(self):
        count = 0
        while True:
            count += 1
            temp_filename = f'temp_image{count}.png'
            temp_path = os.path.join(self.temp_file_directory, temp_filename)
            if not os.path.exists(temp_path):
                return temp_path

    def insert_image(self, image, path):
        if image.width() > self.width():
            image = image.scaledToWidth(self.width())
        cursor = self.textCursor()
        image_format = QTextImageFormat()
        image_format.setName(path)
        image_format.setWidth(image.width())
        image_format.setHeight(image.height())
        cursor.insertImage(image_format)
        QTimer.singleShot(100, lambda: replace_textedit_with_imagetextedit(self))

    def insert_image_from_url(self, url):
        if url.isLocalFile():
            image = QImage(url.toLocalFile())
            self.insert_image(image, url.toLocalFile())
        else:
            response = requests.get(url.toString())
            if response.status_code == 200:
                image_data = response.content
                image = QImage()
                image.loadFromData(image_data)
                temp_path = self.generate_temp_image_path()
                image.save(temp_path)
                self.insert_image(image, temp_path)
            else:
                print(f"Failed to load image from URL: {url.toString()}")

    def cleanup_temp_images(self):
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
        if event.mimeData().hasImage() or event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event):
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
    parent_widget = textedit.parentWidget()
    if not parent_widget:
        raise ValueError("The textedit widget has no parent widget.")

    # Create a new ImageTextEdit instance
    image_textedit = ImageTextEdit(parent_widget)

    # Copy properties from the existing QTextEdit
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

    # Replace the old QTextEdit with the new ImageTextEdit
    layout = parent_widget.layout()
    if layout:
        index = layout.indexOf(textedit)
        if index != -1:
            layout.replaceWidget(textedit, image_textedit)
        else:
            raise ValueError("The textedit widget is not found in the layout.")
    else:
        parent_widget.setLayout(QVBoxLayout())
        parent_widget.layout().addWidget(image_textedit)

    # Update references in the parent widget based on object name
    object_name = textedit.objectName()
    for child in parent_widget.findChildren(QTextEdit):
        if child.objectName() == object_name:
            child.setParent(None)  # Remove old reference
            break

    return image_textedit