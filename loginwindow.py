import os
from datetime import timedelta

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import QMainWindow

from UI.loginUI import Ui_MainWindow
from common import *
from resetwindow import ResetWindow
from toolUI.TipsMessageBox import TipsMessageBox
import re


class LoginWindow(QMainWindow, Ui_MainWindow):
    login_successful = pyqtSignal()

    def __init__(self, parent=None):
        super(LoginWindow, self).__init__(parent)
        self.reset_window = None
        self.m_flag = False
        self.setupUi(self)

        self.setWindowFlags(Qt.FramelessWindowHint)

        self.load_settings()

        self.loginButton.clicked.connect(self.login_user)
        self.signButton.clicked.connect(self.sign_user)
        self.resendButton.clicked.connect(self.resend_verification_code)
        self.resendButton.setDisabled(False)
        self.foggotButton.clicked.connect(self.open_reset_window)

        self.verification_code = None
        self.verification_code_expiry = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.enable_resend_button)

    def login_user(self):
        user_input = self.username_login_in.text()
        password = self.password_login_in.text()

        connection = connect_db()
        cursor = connection.cursor()

        try:
            if "@" in user_input:
                cursor.execute("SELECT email, password FROM users")
                results = cursor.fetchall()
                for encrypted_email, db_password in results:
                    if verify_data(encrypted_email, user_input) and verify_data(db_password, password):
                        self.show_tips("登录成功！")
                        self.login_successful.emit()
                        self.close()
                        return
            else:
                cursor.execute("SELECT password FROM users WHERE username = %s", (user_input,))
                password_result = cursor.fetchone()
                if password_result and verify_data(password_result[0], password):
                    self.show_tips("登录成功！")
                    if self.autologin.isChecked():
                        self.save_settings()
                    self.login_successful.emit()
                    self.close()
                    return
            self.show_tips("用户名/邮箱或密码错误！")
        except pymysql.MySQLError as e:
            self.show_tips(f"数据库错误: 发生错误: {e}")
        finally:
            cursor.close()
            connection.close()

    def sign_user(self):
        username = self.username_sign_up.text()
        password = self.password_sign_up.text()
        email = self.bindemail.text()
        input_code = self.CAPTCHA.text()

        if not (username and password and email and input_code):
            self.show_tips("所有字段均为必填项")
            return

        if len(username) > 16:
            self.show_tips("用户名不能超过16个字符")
            return

        if not re.match("^[a-zA-Z0-9_]+$", username):
            self.show_tips("用户名不能包含特殊字符")
            return

        if not is_password_strong(password):
            self.show_tips("密码必须至少包含8个字符，并包括大写字母、小写字母、数字、特殊符号中的三种")
            return

        if self.verification_code is None or self.verification_code_expiry < get_server_time():
            self.show_tips("验证码无效或已过期，请重新发送验证码")
            return

        if input_code != self.verification_code:
            self.show_tips("验证码错误")
            return

        connection = connect_db()
        cursor = connection.cursor()

        try:
            cursor.execute("SELECT username, email FROM users")
            results = cursor.fetchall()
            for result in results:
                if verify_data(result[1], email):
                    self.show_tips("邮箱已被注册")
                    return
                elif result[0] == username:
                    self.show_tips("用户名已存在")
            else:
                hashed_password = hash_data(password)
                encrypted_email = hash_data(email)
                cursor.execute("INSERT INTO users (username, password, email) VALUES (%s, %s, %s)",
                               (username, hashed_password, encrypted_email))
                connection.commit()
                self.show_tips("注册成功！")
        except pymysql.MySQLError as e:
            self.show_tips(f"数据库错误: 发生错误: {e}")
        finally:
            cursor.close()
            connection.close()

    def resend_verification_code(self):
        email = self.bindemail.text()

        if not email:
            self.show_tips("请先填写邮箱地址")
            return

        self.verification_code = generate_verification_code()
        self.verification_code_expiry = get_server_time() + timedelta(minutes=3)

        send_email(email, "验证码", f"您的验证码是: {self.verification_code}")

        self.resendButton.setDisabled(True)
        self.resendButton.setToolTip("60秒后可重新发送")
        self.timer.start(60000)

    def enable_resend_button(self):
        self.resendButton.setDisabled(False)

    def open_reset_window(self):
        self.reset_window = ResetWindow()
        self.reset_window.show()
        self.hide()  # 隐藏当前登录窗口

        self.reset_window.finished.connect(self.on_reset_window_closed)

    def on_reset_window_closed(self):
        self.show()  # 显示登录窗口
        self.username_login_in.clear()
        self.password_login_in.clear()
        self.username_sign_up.clear()
        self.password_sign_up.clear()
        self.bindemail.clear()
        self.CAPTCHA.clear()
        self.reset_window = None

    def load_settings(self):
        """加载配置文件并设置界面控件"""
        if os.path.exists("config/setting.json"):
            with open("config/setting.json", "r") as file:
                settings = json.load(file)
                self.autologin.setChecked(settings.get("autologin", False))
                if self.autologin.isChecked():
                    self.username_login_in.setText(settings.get("username", ""))
                    self.password_login_in.setText(settings.get("password", ""))

    def save_settings(self):
        """保存当前用户名、密码及自动登录状态"""
        if os.path.exists("config/setting.json"):
            with open("config/setting.json", 'r') as file:
                settings = json.load(file)
        else:
            settings = {}
        settings['username'] = self.username_login_in.text()
        settings['password'] = self.password_login_in.text()
        settings['autologin'] = self.autologin.isChecked()

        with open("config/setting.json", 'w') as file:
            json.dump(settings, file, indent=4)

    def show_tips(self, text):
        """
        显示提示信息
        """
        TipsMessageBox(self.closeButton, title='Tips', text=text, time=2000, auto=True).exec_()

    def mousePressEvent(self, event):
        """
        鼠标按下事件处理。
        """
        self.m_Position = event.pos()
        if event.button() == Qt.LeftButton:
            if 0 < self.m_Position.x() < self.loginBox_body.pos().x() + self.loginBox_body.width() and \
                    0 < self.m_Position.y() < self.loginBox_body.pos().y() + self.loginBox_body.height():
                self.m_flag = True

    def mouseMoveEvent(self, QMouseEvent):
        """
        鼠标移动事件处理。
        """
        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)

    def mouseReleaseEvent(self, QMouseEvent):
        """
        鼠标释放事件处理。
        """
        self.m_flag = False
