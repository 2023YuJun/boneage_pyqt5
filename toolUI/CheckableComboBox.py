from PyQt5 import QtWidgets, QtGui, QtCore


class CheckableComboBox(QtWidgets.QComboBox):
    """
    一个可选多选的QComboBox子类，通过复选框来选择选项。

    参数：
        parent (QWidget): 父控件，默认为None。
    """

    def __init__(self, parent=None):
        super(CheckableComboBox, self).__init__(parent)
        # 设置模型，用于存储列表项
        self.setModel(QtGui.QStandardItemModel(self))
        # 连接按压事件与处理函数
        self.view().pressed.connect(self.handleItemPressed)
        # 多选模式标志
        self.multi_select = False
        # 保存被选中的选项名
        self.selected_items = []

    def handleItemPressed(self, index):
        """
        处理列表项按压事件，更新其选中状态。

        参数：
            index (QModelIndex): 被按压项的索引。
        """
        item = self.model().itemFromIndex(index)
        if self.multi_select:
            # 多选模式下，切换选中状态
            if item.checkState() == QtCore.Qt.Checked:
                item.setCheckState(QtCore.Qt.Unchecked)
                self.selected_items.remove(item.text())
            else:
                item.setCheckState(QtCore.Qt.Checked)
                self.selected_items.append(item.text())
        else:
            # 单选模式下，只选中当前项
            self.selected_items.clear()
            for i in range(self.count()):
                self.model().item(i).setCheckState(QtCore.Qt.Unchecked)
            item.setCheckState(QtCore.Qt.Checked)
            self.selected_items.append(item.text())

        # 更新显示的文本
        self.updateText()

    def updateText(self):
        """
        更新组合框显示的文本，以选中的项的前缀集合来显示。
        """
        if not self.multi_select:
            return

        prefix_set = set()
        for item_text in self.selected_items:
            prefix = item_text.rsplit('-', 1)[0]  # 提取前缀
            prefix_set.add(prefix)

        # 设置组合框的显示文本
        if prefix_set:
            self.setEditText('&'.join(prefix_set))
        else:
            self.setEditText("")

    def addItem(self, text):
        """
        添加单个项，并设置其可选中状态。

        参数：
            text (str): 要添加的项的文本。
        """
        item = QtGui.QStandardItem(text)
        item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
        item.setData(QtCore.Qt.Unchecked, QtCore.Qt.CheckStateRole)
        self.model().appendRow(item)

    def addItems(self, texts):
        """
        批量添加多个项。

        参数：
            texts (list): 要添加的项的文本列表。
        """
        for text in texts:
            self.addItem(text)

    def setMultiSelect(self, multi_select):
        """
        设置组合框是否为多选模式。

        参数：
            multi_select (bool): True表示多选模式，False表示单选模式。
        """
        self.multi_select = multi_select


def replace_comboBox_with_checkable(comboBox):
    """
    用 CheckableComboBox 替换现有的 QComboBox，并保留原来的属性和布局。

    参数：
        comboBox (QComboBox): 要替换的原始 QComboBox。

    返回：
        checkableComboBox (CheckableComboBox): 替换后的 CheckableComboBox 实例。
    """
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
