import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QSplitter, QTextEdit, QLabel

class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()

        # 创建主窗口和布局管理器
        self.splitter = QSplitter(self)
        self.setCentralWidget(self.splitter)

        # 创建左侧和右侧控件
        left_widget = QTextEdit()
        right_widget = QLabel("Right Widget")

        # 添加左侧和右侧控件到分隔条
        self.splitter.addWidget(left_widget)
        self.splitter.addWidget(right_widget)

        # 设置分隔条的初始位置
        self.splitter.setSizes([200, 400])

        # 设置主窗口的一些属性
        self.setWindowTitle("Splitter Example")
        self.setGeometry(100, 100, 800, 600)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
