import sys
from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtWidgets import QApplication
from mainwindow import MainWindow
from loginwindow import LoginWindow


class EmittingStream(QObject):
    textWritten = pyqtSignal(str)

    def write(self, text):
        if text.strip():
            self.textWritten.emit(str(text))

    def flush(self):
        pass


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # 创建LoginForm实例
    LoginForm = LoginWindow()

    def on_login_success():
        # 登录成功后创建并显示主窗口
        MainForm = MainWindow()
        # 重定向 stdout 到 GUI
        sys.stdout = EmittingStream()
        sys.stdout.textWritten.connect(MainForm.bottom_msg)
        MainForm.show()


    # 连接登录成功信号与槽函数
    LoginForm.login_successful.connect(on_login_success)

    # 显示登录窗口
    LoginForm.show()

    sys.exit(app.exec_())
    # app = QApplication(sys.argv)
    # window = MainWindow()
    # window.show()
    # sys.exit(app.exec_())