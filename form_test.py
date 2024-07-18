import os
import sys
from PyQt5 import QtWidgets, QtGui, QtCore


class CheckableComboBox(QtWidgets.QComboBox):
    def __init__(self, parent=None):
        super(CheckableComboBox, self).__init__(parent)
        self.setModel(QtGui.QStandardItemModel(self))
        self.view().pressed.connect(self.handleItemPressed)
        self.multi_select = False
        self.selected_files = []

    def handleItemPressed(self, index):
        if not self.multi_select:
            return
        item = self.model().itemFromIndex(index)
        if item.checkState() == QtCore.Qt.Checked:
            item.setCheckState(QtCore.Qt.Unchecked)
        else:
            item.setCheckState(QtCore.Qt.Checked)
        self.updateText()
        self.updateSelectedFiles()

    def updateText(self):
        text_list = []
        for i in range(self.count()):
            item = self.model().item(i)
            if item.checkState() == QtCore.Qt.Checked:
                text_list.append(item.text())
        self.setEditText(", ".join(text_list))

    def addItem(self, text):
        item = QtGui.QStandardItem(text)
        item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
        item.setData(QtCore.Qt.Unchecked, QtCore.Qt.CheckStateRole)
        self.model().appendRow(item)

    def addItems(self, texts):
        for text in texts:
            self.addItem(text)

    def setMultiSelect(self, multi_select):
        self.multi_select = multi_select

    def updateSelectedFiles(self):
        self.selected_files = []
        for i in range(self.count()):
            item = self.model().item(i)
            if item.checkState() == QtCore.Qt.Checked:
                self.selected_files.append(os.path.join('./model', item.text()))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Model Selector")

        self.comboBox_task = QtWidgets.QComboBox(self)
        self.comboBox_model = CheckableComboBox(self)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.comboBox_task)
        layout.addWidget(self.comboBox_model)

        container = QtWidgets.QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # 自动更新模型
        self.comboBox_task.addItems(["骨龄评估", "检测", "分类"])
        self.model_list = []
        self.update_model_list()

        self.comboBox_task.currentTextChanged.connect(self.update_model_items)
        self.update_model_items()

        self.qtimer_search = QtCore.QTimer(self)
        self.qtimer_search.timeout.connect(self.update_model_list)
        self.qtimer_search.start(1000)  # 更新模型文件选项

    def update_model_list(self):
        model_list = os.listdir('./model')
        model_list = [file for file in model_list if file.endswith('.pt')]
        model_list.sort(key=lambda x: os.path.getsize('./model/' + x))

        if model_list != self.model_list:
            self.model_list = model_list
            self.update_model_items()

    def update_model_items(self):
        selected_task = self.comboBox_task.currentText()
        self.comboBox_model.clear()

        if selected_task == "骨龄评估":
            cls_models = [file for file in self.model_list if file.endswith('-cls.pt')]
            det_models = [file for file in self.model_list if file.endswith('-det.pt')]
            self.comboBox_model.setMultiSelect(True)
            self.comboBox_model.addItems(cls_models + det_models)
        elif selected_task == "检测":
            filtered_list = [file for file in self.model_list if file.endswith('-det.pt')]
            self.comboBox_model.setMultiSelect(False)
            self.comboBox_model.addItems(filtered_list)
        elif selected_task == "分类":
            filtered_list = [file for file in self.model_list if file.endswith('-cls.pt')]
            self.comboBox_model.setMultiSelect(False)
            self.comboBox_model.addItems(filtered_list)

        # 更新选中项的文件路径
        self.comboBox_model.updateSelectedFiles()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
