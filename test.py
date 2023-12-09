import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QSpinBox, QSlider, QWidget

class SpinBoxSliderWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("SpinBox and Slider")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.spin_box = QSpinBox()
        self.slider = QSlider()

        # 设置取值范围
        min_value = 0
        max_value = 100
        self.spin_box.setRange(min_value, max_value)
        self.slider.setRange(min_value, max_value)

        # 初始值设定
        initial_value = 50
        self.spin_box.setValue(initial_value)
        self.slider.setValue(initial_value)

        # 将信号和槽连接起来
        self.spin_box.valueChanged.connect(self.slider.setValue)
        self.slider.valueChanged.connect(self.spin_box.setValue)

        self.layout.addWidget(self.spin_box)
        self.layout.addWidget(self.slider)

def main():
    app = QApplication(sys.argv)
    window = SpinBoxSliderWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
