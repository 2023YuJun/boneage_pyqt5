from PyQt5 import QtWidgets, QtGui, QtCore


class CheckableComboBox(QtWidgets.QComboBox):
    def __init__(self, parent=None):
        super(CheckableComboBox, self).__init__(parent)
        self.setModel(QtGui.QStandardItemModel(self))
        self.view().pressed.connect(self.handleItemPressed)
        self.multi_select = False
        self.selected_files = []

    def handleItemPressed(self, index):
        item = self.model().itemFromIndex(index)
        if self.multi_select:
            if item.checkState() == QtCore.Qt.Checked:
                item.setCheckState(QtCore.Qt.Unchecked)
            else:
                item.setCheckState(QtCore.Qt.Checked)
        else:
            for i in range(self.count()):
                self.model().item(i).setCheckState(QtCore.Qt.Unchecked)
            item.setCheckState(QtCore.Qt.Checked)

        self.updateText()
        self.updateSelectedFiles()

    def updateText(self):
        if not self.multi_select:
            return

        prefix_set = set()
        for i in range(self.count()):
            item = self.model().item(i)
            if item.checkState() == QtCore.Qt.Checked:
                file_name = item.text()
                prefix = file_name.rsplit('-', 1)[0]  # 提取前缀
                prefix_set.add(prefix)

        if prefix_set:
            self.setEditText('&'.join(prefix_set))
        else:
            self.setEditText("")

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
                self.selected_files.append(item.text())
        self.updateText()


def replace_comboBox_with_checkable(comboBox):
    # 复制原有 comboBox 的所有属性
    checkableComboBox = CheckableComboBox(comboBox.parentWidget())
    checkableComboBox.setGeometry(comboBox.geometry())
    checkableComboBox.setObjectName(comboBox.objectName())
    checkableComboBox.setStyleSheet(comboBox.styleSheet())
    checkableComboBox.setFont(comboBox.font())
    checkableComboBox.setSizePolicy(comboBox.sizePolicy())
    checkableComboBox.setMinimumSize(comboBox.minimumSize())
    checkableComboBox.setMaximumSize(comboBox.maximumSize())
    checkableComboBox.setEditable(comboBox.isEditable())

    # 获取 comboBox 在布局中的索引
    layout = comboBox.parentWidget().layout()
    index = layout.indexOf(comboBox)

    # 移除原来的 comboBox
    layout.removeWidget(comboBox)
    comboBox.deleteLater()

    # 将新的 checkableComboBox 添加到原位置
    layout.insertWidget(index, checkableComboBox)

    return checkableComboBox
