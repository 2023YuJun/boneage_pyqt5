import sys

from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtWidgets import QApplication
from mainwindow import MainWindow


class EmittingStream(QObject):
    """
    自定义 stdout 重定向类。

    Attributes:
        textWritten (pyqtSignal): 信号，用于发送重定向的文本。
    """

    textWritten = pyqtSignal(str)

    def write(self, text):
        if text.strip():
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
