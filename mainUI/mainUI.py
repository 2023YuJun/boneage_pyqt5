# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1250, 700)
        MainWindow.setStyleSheet("#mainWindow{\n"
"border:none;\n"
"}\n"
"\n"
"QGroupBox{\n"
"border:1px solid #bababa;  /*边框样式*/\n"
"background-color:#6e6e6e;    /*背景色*/\n"
"}\n"
"\n"
"QPushButton{\n"
"font-family: \"Microsoft YaHei\";\n"
"font-size: 14px;\n"
"font-weight: bold;\n"
"color:white;\n"
"text-align: center center;\n"
"padding-left: 5px;\n"
"padding-right: 5px;\n"
"padding-top: 4px;\n"
"padding-bottom: 4px;\n"
"border-style: solid;\n"
"border-width: 0px;\n"
"border-color: rgba(255, 255, 255, 255);\n"
"border-radius: 3px;\n"
"background-color: rgba(200, 200, 200,0);\n"
"}\n"
"\n"
"/* QPushButton 获得焦点时的样式 */\n"
"QPushButton:focus {\n"
"    outline: none;  /* 去除获得焦点时的默认轮廓线 */\n"
"}\n"
"\n"
"/* QPushButton 鼠标悬停时的样式 */\n"
"QPushButton::hover {\n"
"    border-style: solid;  /* 边框样式 */\n"
"    border-width: 0px;  /* 边框宽度 */\n"
"    border-radius: 0px;  /* 边框圆角 */\n"
"    background-color: rgb(165, 165, 165);  /* 鼠标悬停时的背景色 */\n"
"}\n"
"\n"
"QLabel{\n"
"font-family: \"Microsoft YaHei\";  /* 字体族 */\n"
"    font-weight: bold;  /* 字体粗细 */\n"
"    color: white;  /* 文本颜色 */\n"
"    text-align: center center;  /* 文本居中对齐 */\n"
"    indent:0;\n"
"}\n"
"")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox_body = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_body.setStyleSheet("")
        self.groupBox_body.setTitle("")
        self.groupBox_body.setObjectName("groupBox_body")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.groupBox_body)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setSpacing(0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.groupBox_top = QtWidgets.QGroupBox(self.groupBox_body)
        self.groupBox_top.setMinimumSize(QtCore.QSize(0, 40))
        self.groupBox_top.setMaximumSize(QtCore.QSize(16777215, 40))
        self.groupBox_top.setStyleSheet("#groupBox_top{\n"
"border:none;\n"
"}")
        self.groupBox_top.setTitle("")
        self.groupBox_top.setObjectName("groupBox_top")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.groupBox_top)
        self.horizontalLayout_2.setContentsMargins(10, 0, 5, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label = QtWidgets.QLabel(self.groupBox_top)
        self.label.setStyleSheet("font-size:26px;")
        self.label.setIndent(0)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.minButton = QtWidgets.QPushButton(self.groupBox_top)
        self.minButton.setMinimumSize(QtCore.QSize(0, 30))
        self.minButton.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/images/icon/最小化.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.minButton.setIcon(icon)
        self.minButton.setObjectName("minButton")
        self.horizontalLayout_3.addWidget(self.minButton)
        self.maxButton = QtWidgets.QPushButton(self.groupBox_top)
        self.maxButton.setMinimumSize(QtCore.QSize(0, 30))
        self.maxButton.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/images/icon/正方形.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon1.addPixmap(QtGui.QPixmap(":/images/icon/还原.png"), QtGui.QIcon.Active, QtGui.QIcon.On)
        icon1.addPixmap(QtGui.QPixmap(":/images/icon/还原.png"), QtGui.QIcon.Selected, QtGui.QIcon.On)
        self.maxButton.setIcon(icon1)
        self.maxButton.setObjectName("maxButton")
        self.horizontalLayout_3.addWidget(self.maxButton)
        self.closeButton = QtWidgets.QPushButton(self.groupBox_top)
        self.closeButton.setMinimumSize(QtCore.QSize(0, 30))
        self.closeButton.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/images/icon/关闭.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.closeButton.setIcon(icon2)
        self.closeButton.setObjectName("closeButton")
        self.horizontalLayout_3.addWidget(self.closeButton)
        self.horizontalLayout_2.addLayout(self.horizontalLayout_3)
        self.verticalLayout_4.addWidget(self.groupBox_top)
        self.groupBox_fill = QtWidgets.QGroupBox(self.groupBox_body)
        self.groupBox_fill.setStyleSheet("")
        self.groupBox_fill.setTitle("")
        self.groupBox_fill.setObjectName("groupBox_fill")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.groupBox_fill)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.groupBox_sidebar = QtWidgets.QGroupBox(self.groupBox_fill)
        self.groupBox_sidebar.setMaximumSize(QtCore.QSize(350, 16777215))
        self.groupBox_sidebar.setStyleSheet("#groupBox_sidebar{\n"
"border:none;\n"
"border-right:1px solid #bababa;\n"
"}")
        self.groupBox_sidebar.setTitle("")
        self.groupBox_sidebar.setObjectName("groupBox_sidebar")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.groupBox_sidebar)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.groupBox = QtWidgets.QGroupBox(self.groupBox_sidebar)
        self.groupBox.setMinimumSize(QtCore.QSize(0, 40))
        self.groupBox.setMaximumSize(QtCore.QSize(16777215, 40))
        self.groupBox.setStyleSheet("border:none;")
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.groupBox)
        self.horizontalLayout_6.setContentsMargins(10, 0, 10, 0)
        self.horizontalLayout_6.setSpacing(0)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setStyleSheet("font-size:22px;")
        self.label_3.setIndent(0)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_6.addWidget(self.label_3)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem1)
        self.verticalLayout_3.addWidget(self.groupBox)
        self.groupBox_2 = QtWidgets.QGroupBox(self.groupBox_sidebar)
        self.groupBox_2.setMinimumSize(QtCore.QSize(0, 30))
        self.groupBox_2.setMaximumSize(QtCore.QSize(16777215, 30))
        self.groupBox_2.setStyleSheet("#groupBox_2{\n"
"border:none;\n"
"border-top:1px solid #bababa;\n"
"}\n"
"")
        self.groupBox_2.setTitle("")
        self.groupBox_2.setObjectName("groupBox_2")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.groupBox_2)
        self.horizontalLayout_7.setContentsMargins(10, 0, 20, 0)
        self.horizontalLayout_7.setSpacing(0)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_6 = QtWidgets.QLabel(self.groupBox_2)
        self.label_6.setMinimumSize(QtCore.QSize(0, 0))
        self.label_6.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.label_6.setStyleSheet("font-size:18px;")
        self.label_6.setIndent(0)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_7.addWidget(self.label_6)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem2)
        self.comboBox_task = QtWidgets.QComboBox(self.groupBox_2)
        self.comboBox_task.setStyleSheet("QComboBox{\n"
"font-family: \"Microsoft YaHei\";\n"
"font-size: 16px;\n"
"color: rgb(218, 218, 218);\n"
"border-width:0px;\n"
"border-color:white;\n"
"border-style:solid;\n"
"background-color: rgba(200, 200, 200,0);}\n"
"\n"
"")
        self.comboBox_task.setObjectName("comboBox_task")
        self.horizontalLayout_7.addWidget(self.comboBox_task)
        self.comboBox_model = QtWidgets.QComboBox(self.groupBox_2)
        self.comboBox_model.setStyleSheet("QComboBox{\n"
"font-family: \"Microsoft YaHei\";\n"
"font-size: 16px;\n"
"color: rgb(218, 218, 218);\n"
"border-width:0px;\n"
"border-color:white;\n"
"border-style:solid;\n"
"background-color: rgba(200, 200, 200,0);}\n"
"\n"
"")
        self.comboBox_model.setObjectName("comboBox_model")
        self.horizontalLayout_7.addWidget(self.comboBox_model)
        self.verticalLayout_3.addWidget(self.groupBox_2)
        self.groupBox_3 = QtWidgets.QGroupBox(self.groupBox_sidebar)
        self.groupBox_3.setMinimumSize(QtCore.QSize(0, 30))
        self.groupBox_3.setMaximumSize(QtCore.QSize(16777215, 30))
        self.groupBox_3.setStyleSheet("#groupBox_3{\n"
"border:none;\n"
"}")
        self.groupBox_3.setTitle("")
        self.groupBox_3.setObjectName("groupBox_3")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout(self.groupBox_3)
        self.horizontalLayout_9.setContentsMargins(10, 0, 20, 0)
        self.horizontalLayout_9.setSpacing(0)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.label_7 = QtWidgets.QLabel(self.groupBox_3)
        self.label_7.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.label_7.setStyleSheet("font-size:18px;")
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_9.addWidget(self.label_7)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_9.addItem(spacerItem3)
        self.groupBox_input = QtWidgets.QGroupBox(self.groupBox_3)
        self.groupBox_input.setStyleSheet("#groupBox_input{\n"
"border:none;\n"
"}")
        self.groupBox_input.setTitle("")
        self.groupBox_input.setObjectName("groupBox_input")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.groupBox_input)
        self.horizontalLayout_8.setContentsMargins(5, 0, 5, 0)
        self.horizontalLayout_8.setSpacing(5)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.fileButton = QtWidgets.QPushButton(self.groupBox_input)
        self.fileButton.setText("")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/images/icon/打开.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.fileButton.setIcon(icon3)
        self.fileButton.setObjectName("fileButton")
        self.horizontalLayout_8.addWidget(self.fileButton)
        self.cameraButton = QtWidgets.QPushButton(self.groupBox_input)
        self.cameraButton.setText("")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/images/icon/摄像头开.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon4.addPixmap(QtGui.QPixmap(":/images/icon/摄像头关.png"), QtGui.QIcon.Active, QtGui.QIcon.On)
        icon4.addPixmap(QtGui.QPixmap(":/images/icon/摄像头关.png"), QtGui.QIcon.Selected, QtGui.QIcon.On)
        self.cameraButton.setIcon(icon4)
        self.cameraButton.setObjectName("cameraButton")
        self.horizontalLayout_8.addWidget(self.cameraButton)
        self.rtspButton = QtWidgets.QPushButton(self.groupBox_input)
        self.rtspButton.setText("")
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(":/images/icon/实时视频流解析.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.rtspButton.setIcon(icon5)
        self.rtspButton.setObjectName("rtspButton")
        self.horizontalLayout_8.addWidget(self.rtspButton)
        self.horizontalLayout_9.addWidget(self.groupBox_input)
        self.verticalLayout_3.addWidget(self.groupBox_3)
        self.groupBox_4 = QtWidgets.QGroupBox(self.groupBox_sidebar)
        self.groupBox_4.setMinimumSize(QtCore.QSize(0, 60))
        self.groupBox_4.setMaximumSize(QtCore.QSize(16777215, 60))
        self.groupBox_4.setStyleSheet("#groupBox_4{\n"
"border:none;\n"
"}")
        self.groupBox_4.setTitle("")
        self.groupBox_4.setObjectName("groupBox_4")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.groupBox_4)
        self.verticalLayout_6.setContentsMargins(10, 0, 10, 0)
        self.verticalLayout_6.setSpacing(0)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.label_8 = QtWidgets.QLabel(self.groupBox_4)
        self.label_8.setStyleSheet("font-size:18px;")
        self.label_8.setObjectName("label_8")
        self.verticalLayout_6.addWidget(self.label_8)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setContentsMargins(0, -1, 0, -1)
        self.horizontalLayout_10.setSpacing(10)
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.iouSpinBox = QtWidgets.QDoubleSpinBox(self.groupBox_4)
        self.iouSpinBox.setStyleSheet("QDoubleSpinBox{\n"
"background:rgba(200, 200, 200,50);\n"
"color:white;\n"
"font-size: 14px;\n"
"font-family: \"Microsoft YaHei UI\";\n"
"border-style: solid;\n"
"border-width: 1px;\n"
"border-color: rgba(200, 200, 200,100);\n"
"border-radius: 3px;}\n"
"\n"
"QDoubleSpinBox::down-button{\n"
"background:rgba(200, 200, 200,0);\n"
"border-image: url(:/images/icon/箭头_列表展开.png);}\n"
"QDoubleSpinBox::down-button::hover{\n"
"background:rgba(200, 200, 200,100);\n"
"border-image: url(:/images/icon/箭头_列表展开.png);}\n"
"\n"
"QDoubleSpinBox::up-button{\n"
"background:rgba(200, 200, 200,0);\n"
"border-image: url(:/images/icon/箭头_列表收起.png);}\n"
"QDoubleSpinBox::up-button::hover{\n"
"background:rgba(200, 200, 200,100);\n"
"border-image: url(:/images/icon/箭头_列表收起.png);}\n"
"")
        self.iouSpinBox.setMaximum(1.0)
        self.iouSpinBox.setSingleStep(0.01)
        self.iouSpinBox.setProperty("value", 0.4)
        self.iouSpinBox.setObjectName("iouSpinBox")
        self.horizontalLayout_10.addWidget(self.iouSpinBox)
        self.iouSlider = QtWidgets.QSlider(self.groupBox_4)
        self.iouSlider.setStyleSheet("QSlider{\n"
"border-color: #bcbcbc;\n"
"color:#d9d9d9;\n"
"}\n"
"QSlider::groove:horizontal {                                \n"
"     border: 1px solid #999999;                             \n"
"     height: 3px;                                           \n"
"    margin: 0px 0;                                         \n"
"     left: 5px; right: 5px; \n"
" }\n"
"QSlider::handle:horizontal {                               \n"
"     border: 0px ; \n"
"     border-image:url(:/images/icon/圆.png);\n"
"     width:15px;\n"
"     margin: -7px -7px -7px -7px;                  \n"
"} \n"
"QSlider::add-page:horizontal{\n"
"background: qlineargradient(spread:pad, x1:0, y1:1, x2:0, y2:0, stop:0 #d9d9d9, stop:0.25 #d9d9d9, stop:0.5 #d9d9d9, stop:1 #d9d9d9); \n"
"\n"
"}\n"
"QSlider::sub-page:horizontal{                               \n"
" background: qlineargradient(spread:pad, x1:0, y1:1, x2:0, y2:0, stop:0 #373737, stop:0.25 #373737, stop:0.5 #373737, stop:1 #373737);                     \n"
"}")
        self.iouSlider.setMaximum(100)
        self.iouSlider.setProperty("value", 40)
        self.iouSlider.setOrientation(QtCore.Qt.Horizontal)
        self.iouSlider.setObjectName("iouSlider")
        self.horizontalLayout_10.addWidget(self.iouSlider)
        self.verticalLayout_6.addLayout(self.horizontalLayout_10)
        self.verticalLayout_3.addWidget(self.groupBox_4)
        self.groupBox_5 = QtWidgets.QGroupBox(self.groupBox_sidebar)
        self.groupBox_5.setMinimumSize(QtCore.QSize(0, 60))
        self.groupBox_5.setMaximumSize(QtCore.QSize(16777215, 60))
        self.groupBox_5.setStyleSheet("#groupBox_5{\n"
"border:none;\n"
"}")
        self.groupBox_5.setTitle("")
        self.groupBox_5.setObjectName("groupBox_5")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.groupBox_5)
        self.verticalLayout_7.setContentsMargins(10, 0, 10, 0)
        self.verticalLayout_7.setSpacing(0)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.label_10 = QtWidgets.QLabel(self.groupBox_5)
        self.label_10.setStyleSheet("font-size:18px;")
        self.label_10.setObjectName("label_10")
        self.verticalLayout_7.addWidget(self.label_10)
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.confSpinBox = QtWidgets.QDoubleSpinBox(self.groupBox_5)
        self.confSpinBox.setStyleSheet("QDoubleSpinBox{\n"
"background:rgba(200, 200, 200,50);\n"
"color:white;\n"
"font-size: 14px;\n"
"font-family: \"Microsoft YaHei UI\";\n"
"border-style: solid;\n"
"border-width: 1px;\n"
"border-color: rgba(200, 200, 200,100);\n"
"border-radius: 3px;}\n"
"\n"
"QDoubleSpinBox::down-button{\n"
"background:rgba(200, 200, 200,0);\n"
"border-image: url(:/images/icon/箭头_列表展开.png);}\n"
"QDoubleSpinBox::down-button::hover{\n"
"background:rgba(200, 200, 200,100);\n"
"border-image: url(:/images/icon/箭头_列表展开.png);}\n"
"\n"
"QDoubleSpinBox::up-button{\n"
"background:rgba(200, 200, 200,0);\n"
"border-image: url(:/images/icon/箭头_列表收起.png);}\n"
"QDoubleSpinBox::up-button::hover{\n"
"background:rgba(200, 200, 200,100);\n"
"border-image: url(:/images/icon/箭头_列表收起.png);}\n"
"")
        self.confSpinBox.setMaximum(1.0)
        self.confSpinBox.setSingleStep(0.01)
        self.confSpinBox.setProperty("value", 0.2)
        self.confSpinBox.setObjectName("confSpinBox")
        self.horizontalLayout_12.addWidget(self.confSpinBox)
        self.confSlider = QtWidgets.QSlider(self.groupBox_5)
        self.confSlider.setStyleSheet("QSlider{\n"
"border-color: #bcbcbc;\n"
"color:#d9d9d9;\n"
"}\n"
"QSlider::groove:horizontal {                                \n"
"     border: 1px solid #999999;                             \n"
"     height: 3px;                                           \n"
"    margin: 0px 0;                                         \n"
"     left: 5px; right: 5px; \n"
" }\n"
"QSlider::handle:horizontal {                               \n"
"     border: 0px ; \n"
"     border-image:url(:/images/icon/圆.png);\n"
"     width:15px;\n"
"     margin: -7px -7px -7px -7px;                  \n"
"} \n"
"QSlider::add-page:horizontal{\n"
"background: qlineargradient(spread:pad, x1:0, y1:1, x2:0, y2:0, stop:0 #d9d9d9, stop:0.25 #d9d9d9, stop:0.5 #d9d9d9, stop:1 #d9d9d9); \n"
"\n"
"}\n"
"QSlider::sub-page:horizontal{                               \n"
" background: qlineargradient(spread:pad, x1:0, y1:1, x2:0, y2:0, stop:0 #373737, stop:0.25 #373737, stop:0.5 #373737, stop:1 #373737);                     \n"
"}")
        self.confSlider.setMaximum(100)
        self.confSlider.setProperty("value", 20)
        self.confSlider.setOrientation(QtCore.Qt.Horizontal)
        self.confSlider.setObjectName("confSlider")
        self.horizontalLayout_12.addWidget(self.confSlider)
        self.verticalLayout_7.addLayout(self.horizontalLayout_12)
        self.verticalLayout_3.addWidget(self.groupBox_5)
        self.groupBox_6 = QtWidgets.QGroupBox(self.groupBox_sidebar)
        self.groupBox_6.setMinimumSize(QtCore.QSize(0, 60))
        self.groupBox_6.setMaximumSize(QtCore.QSize(16777215, 60))
        self.groupBox_6.setStyleSheet("#groupBox_6{\n"
"border:none;\n"
"}")
        self.groupBox_6.setTitle("")
        self.groupBox_6.setObjectName("groupBox_6")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.groupBox_6)
        self.verticalLayout_8.setContentsMargins(10, 0, 10, 0)
        self.verticalLayout_8.setSpacing(0)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_13.setContentsMargins(-1, -1, -1, 0)
        self.horizontalLayout_13.setSpacing(5)
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.label_11 = QtWidgets.QLabel(self.groupBox_6)
        self.label_11.setMaximumSize(QtCore.QSize(80, 16777215))
        self.label_11.setStyleSheet("font-size:18px;")
        self.label_11.setObjectName("label_11")
        self.horizontalLayout_13.addWidget(self.label_11)
        self.checkBox_enable = QtWidgets.QCheckBox(self.groupBox_6)
        self.checkBox_enable.setStyleSheet("\n"
"QCheckBox\n"
"{font-size: 16px;\n"
"    font-family: \"Microsoft YaHei\";\n"
"    font-weight: bold;\n"
"         border-radius:9px;\n"
"        background:rgba(66, 195, 255, 0);\n"
"color: rgb(218, 218, 218);;}\n"
"\n"
"QCheckBox::indicator {\n"
"    width: 20px;\n"
"    height: 20px;\n"
"}\n"
"QCheckBox::indicator:unchecked {\n"
"    image: url(:/images/icon/button-off.png);\n"
"}\n"
"\n"
"QCheckBox::indicator:checked {\n"
"    \n"
"    image: url(:/images/icon/button-on.png);\n"
"}\n"
"")
        self.checkBox_enable.setObjectName("checkBox_enable")
        self.horizontalLayout_13.addWidget(self.checkBox_enable)
        self.verticalLayout_8.addLayout(self.horizontalLayout_13)
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.rateSpinBox = QtWidgets.QSpinBox(self.groupBox_6)
        self.rateSpinBox.setMinimumSize(QtCore.QSize(51, 0))
        self.rateSpinBox.setStyleSheet("QSpinBox{\n"
"background:rgba(200, 200, 200,50);\n"
"color:white;\n"
"font-size: 14px;\n"
"font-family: \"Microsoft YaHei UI\";\n"
"border-style: solid;\n"
"border-width: 1px;\n"
"border-color: rgba(200, 200, 200,100);\n"
"border-radius: 3px;}\n"
"\n"
"QSpinBox::down-button{\n"
"background:rgba(200, 200, 200,0);\n"
"border-image: url(:/images/icon/箭头_列表展开.png);}\n"
"QDoubleSpinBox::down-button::hover{\n"
"background:rgba(200, 200, 200,100);\n"
"border-image: url(:/images/icon/箭头_列表展开.png);}\n"
"\n"
"QSpinBox::up-button{\n"
"background:rgba(200, 200, 200,0);\n"
"border-image: url(:/images/icon/箭头_列表收起.png);}\n"
"QSpinBox::up-button::hover{\n"
"background:rgba(200, 200, 200,100);\n"
"border-image: url(:/images/icon/箭头_列表收起.png);}\n"
"")
        self.rateSpinBox.setMinimum(1)
        self.rateSpinBox.setMaximum(20)
        self.rateSpinBox.setObjectName("rateSpinBox")
        self.horizontalLayout_14.addWidget(self.rateSpinBox)
        self.rateSlider = QtWidgets.QSlider(self.groupBox_6)
        self.rateSlider.setStyleSheet("QSlider{\n"
"border-color: #bcbcbc;\n"
"color:#d9d9d9;\n"
"}\n"
"QSlider::groove:horizontal {                                \n"
"     border: 1px solid #999999;                             \n"
"     height: 3px;                                           \n"
"    margin: 0px 0;                                         \n"
"     left: 5px; right: 5px; \n"
" }\n"
"QSlider::handle:horizontal {                               \n"
"     border: 0px ; \n"
"     border-image:url(:/images/icon/圆.png);\n"
"     width:15px;\n"
"     margin: -7px -7px -7px -7px;                  \n"
"} \n"
"QSlider::add-page:horizontal{\n"
"background: qlineargradient(spread:pad, x1:0, y1:1, x2:0, y2:0, stop:0 #d9d9d9, stop:0.25 #d9d9d9, stop:0.5 #d9d9d9, stop:1 #d9d9d9); \n"
"\n"
"}\n"
"QSlider::sub-page:horizontal{                               \n"
" background: qlineargradient(spread:pad, x1:0, y1:1, x2:0, y2:0, stop:0 #373737, stop:0.25 #373737, stop:0.5 #373737, stop:1 #373737);                     \n"
"}")
        self.rateSlider.setMaximum(20)
        self.rateSlider.setPageStep(1)
        self.rateSlider.setProperty("value", 1)
        self.rateSlider.setOrientation(QtCore.Qt.Horizontal)
        self.rateSlider.setObjectName("rateSlider")
        self.horizontalLayout_14.addWidget(self.rateSlider)
        self.verticalLayout_8.addLayout(self.horizontalLayout_14)
        self.verticalLayout_3.addWidget(self.groupBox_6)
        self.groupBox_10 = QtWidgets.QGroupBox(self.groupBox_sidebar)
        self.groupBox_10.setMinimumSize(QtCore.QSize(0, 30))
        self.groupBox_10.setMaximumSize(QtCore.QSize(16777215, 30))
        self.groupBox_10.setStyleSheet("#groupBox_10{\n"
"border:none;\n"
"}")
        self.groupBox_10.setTitle("")
        self.groupBox_10.setObjectName("groupBox_10")
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout(self.groupBox_10)
        self.horizontalLayout_15.setContentsMargins(10, 0, 10, 0)
        self.horizontalLayout_15.setSpacing(5)
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.runButton = QtWidgets.QPushButton(self.groupBox_10)
        self.runButton.setText("")
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(":/images/icon/运行.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon6.addPixmap(QtGui.QPixmap(":/images/icon/暂停.png"), QtGui.QIcon.Active, QtGui.QIcon.On)
        icon6.addPixmap(QtGui.QPixmap(":/images/icon/暂停.png"), QtGui.QIcon.Selected, QtGui.QIcon.On)
        self.runButton.setIcon(icon6)
        self.runButton.setObjectName("runButton")
        self.horizontalLayout_15.addWidget(self.runButton)
        self.progressBar = QtWidgets.QProgressBar(self.groupBox_10)
        self.progressBar.setMaximumSize(QtCore.QSize(16777215, 12))
        self.progressBar.setStyleSheet("QProgressBar{ \n"
"color: rgb(255, 255, 255); \n"
"font:10px; \n"
"border-radius:5px; \n"
"text-align:center; \n"
"border:none; \n"
"background-color: rgba(215, 215, 215,100);\n"
"} \n"
"QProgressBar:chunk{ \n"
"border-radius:5px; \n"
"background: rgba(55, 55, 55, 200);\n"
"}")
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.horizontalLayout_15.addWidget(self.progressBar)
        self.stopButton = QtWidgets.QPushButton(self.groupBox_10)
        self.stopButton.setText("")
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap(":/images/icon/终止.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.stopButton.setIcon(icon7)
        self.stopButton.setObjectName("stopButton")
        self.horizontalLayout_15.addWidget(self.stopButton)
        self.verticalLayout_3.addWidget(self.groupBox_10)
        self.groupBox_11 = QtWidgets.QGroupBox(self.groupBox_sidebar)
        self.groupBox_11.setMinimumSize(QtCore.QSize(0, 40))
        self.groupBox_11.setMaximumSize(QtCore.QSize(16777215, 40))
        self.groupBox_11.setStyleSheet("#groupBox_11{\n"
"border:none;\n"
"}")
        self.groupBox_11.setTitle("")
        self.groupBox_11.setObjectName("groupBox_11")
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout(self.groupBox_11)
        self.horizontalLayout_16.setContentsMargins(10, 5, 10, 5)
        self.horizontalLayout_16.setSpacing(0)
        self.horizontalLayout_16.setObjectName("horizontalLayout_16")
        self.checkBox_autosave = QtWidgets.QCheckBox(self.groupBox_11)
        self.checkBox_autosave.setStyleSheet("\n"
"QCheckBox\n"
"{font-size: 16px;\n"
"    font-family: \"Microsoft YaHei\";\n"
"    font-weight: bold;\n"
"         border-radius:9px;\n"
"        background:rgba(66, 195, 255, 0);\n"
"color: rgb(218, 218, 218);;}\n"
"\n"
"QCheckBox::indicator {\n"
"    width: 20px;\n"
"    height: 20px;\n"
"}\n"
"QCheckBox::indicator:unchecked {\n"
"    image: url(:/images/icon/button-off.png);\n"
"}\n"
"\n"
"QCheckBox::indicator:checked {\n"
"    \n"
"    image: url(:/images/icon/button-on.png);\n"
"}\n"
"")
        self.checkBox_autosave.setObjectName("checkBox_autosave")
        self.horizontalLayout_16.addWidget(self.checkBox_autosave)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_16.addItem(spacerItem4)
        self.horizontalLayout_18 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_18.setSpacing(5)
        self.horizontalLayout_18.setObjectName("horizontalLayout_18")
        self.pathButton = QtWidgets.QPushButton(self.groupBox_11)
        self.pathButton.setStyleSheet("#pathButton{\n"
"border:1px solid #bababa;\n"
"}")
        self.pathButton.setObjectName("pathButton")
        self.horizontalLayout_18.addWidget(self.pathButton)
        self.saveButton = QtWidgets.QPushButton(self.groupBox_11)
        self.saveButton.setStyleSheet("#saveButton{\n"
"border:1px solid #bababa;\n"
"}")
        self.saveButton.setObjectName("saveButton")
        self.horizontalLayout_18.addWidget(self.saveButton)
        self.horizontalLayout_16.addLayout(self.horizontalLayout_18)
        self.verticalLayout_3.addWidget(self.groupBox_11)
        self.groupBox_12 = QtWidgets.QGroupBox(self.groupBox_sidebar)
        self.groupBox_12.setMinimumSize(QtCore.QSize(0, 35))
        self.groupBox_12.setMaximumSize(QtCore.QSize(16777215, 35))
        self.groupBox_12.setStyleSheet("#groupBox_12{\n"
"border:none;\n"
"border-top:1px solid #bababa;\n"
"border-bottom:1px solid #bababa;\n"
"}")
        self.groupBox_12.setTitle("")
        self.groupBox_12.setObjectName("groupBox_12")
        self.horizontalLayout_17 = QtWidgets.QHBoxLayout(self.groupBox_12)
        self.horizontalLayout_17.setContentsMargins(10, 0, 10, 0)
        self.horizontalLayout_17.setSpacing(0)
        self.horizontalLayout_17.setObjectName("horizontalLayout_17")
        self.label_12 = QtWidgets.QLabel(self.groupBox_12)
        self.label_12.setStyleSheet("font-size:22px;")
        self.label_12.setObjectName("label_12")
        self.horizontalLayout_17.addWidget(self.label_12)
        spacerItem5 = QtWidgets.QSpacerItem(254, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_17.addItem(spacerItem5)
        self.verticalLayout_3.addWidget(self.groupBox_12)
        self.groupBox_13 = QtWidgets.QGroupBox(self.groupBox_sidebar)
        self.groupBox_13.setStyleSheet("#groupBox_13{\n"
"border:none;\n"
"}")
        self.groupBox_13.setTitle("")
        self.groupBox_13.setObjectName("groupBox_13")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.groupBox_13)
        self.verticalLayout_9.setContentsMargins(10, 0, 10, 0)
        self.verticalLayout_9.setSpacing(0)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.listWidget = QtWidgets.QListWidget(self.groupBox_13)
        self.listWidget.setStyleSheet("QListWidget{\n"
"background-color: rgba(12, 28, 77, 0);\n"
"\n"
"border-radius:0px;\n"
"font-family: \"Microsoft YaHei\";\n"
"font-size: 16px;\n"
"color: rgb(218, 218, 218);\n"
"}\n"
"")
        self.listWidget.setObjectName("listWidget")
        self.verticalLayout_9.addWidget(self.listWidget)
        self.verticalLayout_3.addWidget(self.groupBox_13)
        self.horizontalLayout.addWidget(self.groupBox_sidebar)
        self.groupBox_view = QtWidgets.QGroupBox(self.groupBox_fill)
        self.groupBox_view.setStyleSheet("#groupBox_view{\n"
"background-color:#8b8b8b;\n"
"border:none;\n"
"}")
        self.groupBox_view.setTitle("")
        self.groupBox_view.setObjectName("groupBox_view")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.groupBox_view)
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_5.setSpacing(0)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.groupBox_7 = QtWidgets.QGroupBox(self.groupBox_view)
        self.groupBox_7.setMinimumSize(QtCore.QSize(0, 40))
        self.groupBox_7.setMaximumSize(QtCore.QSize(16777215, 40))
        self.groupBox_7.setStyleSheet("background-color: rgba(200, 200, 200,0);\n"
"border:none;")
        self.groupBox_7.setTitle("")
        self.groupBox_7.setObjectName("groupBox_7")
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout(self.groupBox_7)
        self.horizontalLayout_11.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_11.setSpacing(0)
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.label_9 = QtWidgets.QLabel(self.groupBox_7)
        self.label_9.setStyleSheet("font-size:22px;")
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_11.addWidget(self.label_9)
        spacerItem6 = QtWidgets.QSpacerItem(857, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_11.addItem(spacerItem6)
        self.verticalLayout_5.addWidget(self.groupBox_7)
        self.groupBox_8 = QtWidgets.QGroupBox(self.groupBox_view)
        self.groupBox_8.setStyleSheet("background-color: rgba(200, 200, 200,0);\n"
"border-left:none;\n"
"border-right:none;")
        self.groupBox_8.setTitle("")
        self.groupBox_8.setObjectName("groupBox_8")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.groupBox_8)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setSpacing(0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_previous = QtWidgets.QLabel(self.groupBox_8)
        self.label_previous.setStyleSheet("border-right:1px solid #bababa;")
        self.label_previous.setText("")
        self.label_previous.setObjectName("label_previous")
        self.horizontalLayout_5.addWidget(self.label_previous)
        self.label_current = QtWidgets.QLabel(self.groupBox_8)
        self.label_current.setStyleSheet("")
        self.label_current.setText("")
        self.label_current.setObjectName("label_current")
        self.horizontalLayout_5.addWidget(self.label_current)
        self.verticalLayout_5.addWidget(self.groupBox_8)
        self.groupBox_9 = QtWidgets.QGroupBox(self.groupBox_view)
        self.groupBox_9.setMaximumSize(QtCore.QSize(16777215, 160))
        self.groupBox_9.setStyleSheet("background-color: rgba(200, 200, 200,0);\n"
"border:none;")
        self.groupBox_9.setTitle("")
        self.groupBox_9.setObjectName("groupBox_9")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout(self.groupBox_9)
        self.verticalLayout_10.setContentsMargins(10, 10, 10, 10)
        self.verticalLayout_10.setSpacing(5)
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.textEdit = QtWidgets.QTextEdit(self.groupBox_9)
        self.textEdit.setObjectName("textEdit")
        self.verticalLayout_10.addWidget(self.textEdit)
        self.verticalLayout_5.addWidget(self.groupBox_9)
        self.horizontalLayout.addWidget(self.groupBox_view)
        self.verticalLayout_4.addWidget(self.groupBox_fill)
        self.groupBox_bottom = QtWidgets.QGroupBox(self.groupBox_body)
        self.groupBox_bottom.setMaximumSize(QtCore.QSize(16777215, 30))
        self.groupBox_bottom.setStyleSheet("#groupBox_bottom{\n"
"border:none;\n"
"}")
        self.groupBox_bottom.setTitle("")
        self.groupBox_bottom.setObjectName("groupBox_bottom")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.groupBox_bottom)
        self.horizontalLayout_4.setContentsMargins(10, 0, 10, 0)
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_bottom = QtWidgets.QLabel(self.groupBox_bottom)
        self.label_bottom.setMinimumSize(QtCore.QSize(0, 30))
        self.label_bottom.setText("")
        self.label_bottom.setObjectName("label_bottom")
        self.horizontalLayout_4.addWidget(self.label_bottom)
        self.verticalLayout_4.addWidget(self.groupBox_bottom)
        self.verticalLayout.addWidget(self.groupBox_body)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.comboBox_task.setCurrentIndex(-1)
        self.closeButton.clicked.connect(MainWindow.close) # type: ignore
        self.minButton.clicked.connect(MainWindow.showMinimized) # type: ignore
        self.maxButton.clicked.connect(MainWindow.showMaximized) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "骨龄评估"))
        self.label_3.setText(_translate("MainWindow", "设置"))
        self.label_6.setText(_translate("MainWindow", "模型"))
        self.label_7.setText(_translate("MainWindow", "输入选择"))
        self.label_8.setText(_translate("MainWindow", "IoU"))
        self.label_10.setText(_translate("MainWindow", "置信度"))
        self.label_11.setText(_translate("MainWindow", "帧间延时"))
        self.checkBox_enable.setText(_translate("MainWindow", "启用"))
        self.checkBox_autosave.setText(_translate("MainWindow", "自动保存"))
        self.pathButton.setText(_translate("MainWindow", "选择保存路径"))
        self.saveButton.setText(_translate("MainWindow", "保存"))
        self.label_12.setText(_translate("MainWindow", "检测情况"))
        self.label_9.setText(_translate("MainWindow", "视图"))
import UIqrc_rc
