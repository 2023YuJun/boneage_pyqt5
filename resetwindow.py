from datetime import timedelta

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import QMainWindow

from UI.resetUI import Ui_MainWindow
from common import *
from toolUI.TipsMessageBox import TipsMessageBox


class ResetWindow(QMainWindow, Ui_MainWindow):
    finished = pyqtSignal()

    def __init__(self, parent=None):
        super(ResetWindow, self).__init__(parent)
        self.setupUi(self)

        self.m_flag = False
        self.setWindowFlags(Qt.FramelessWindowHint)

        self.resetButton.clicked.connect(self.reset_password)
        self.resendButton.clicked.connect(self.resend_verification_code)
        self.resendButton.setDisabled(False)

        self.verification_code = None
        self.verification_code_expiry = None
        self.encrypted_email = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.enable_resend_button)

    def reset_password(self):
        email = self.email.text()
        password = self.password_reset.text()
        confirm_password = self.password_reset_2.text()
        input_code = self.CAPTCHA.text()

        if not email or not password or not confirm_password or not input_code:
            self.show_tips("所有字段均为必填项")
            return

        if password != confirm_password:
            self.show_tips("两次输入的密码不一致")
            return

        if not is_password_strong(password):
            self.show_tips("密码必须至少包含8个字符，并包括大写字母、小写字母、数字、特殊符号中的三种")
            return

        # 检查验证码
        if self.verification_code is None or self.verification_code_expiry < get_server_time():
            self.show_tips("验证码无效或已过期，请重新发送验证码")
            return

        if input_code != self.verification_code:
            self.show_tips("验证码错误")
            return

        connection = connect_db()
        cursor = connection.cursor()

        try:
            # 更新密码
            hashed_password = hash_data(password)
            cursor.execute("UPDATE users SET password = %s WHERE email = %s", (hashed_password, self.encrypted_email))
            connection.commit()
            self.show_tips("密码已重置成功！")
            self.close()

        except pymysql.MySQLError as e:
            self.show_tips(f"数据库错误: 发生错误: {e}")
        finally:
            cursor.close()
            connection.close()

    def resend_verification_code(self):
        email = self.email.text()
        if not email:
            self.show_tips("请先填写邮箱地址")
            return

        connection = connect_db()
        cursor = connection.cursor()

        try:
            # 检查邮箱是否已注册
            cursor.execute("SELECT email FROM users")
            results = cursor.fetchall()

            for result in results:
                if verify_data(result[0], email):
                    self.encrypted_email = result[0]
                    break
            else:
                self.show_tips("邮箱未注册")

            # 生成并发送验证码
            self.verification_code = generate_verification_code()
            self.verification_code_expiry = get_server_time() + timedelta(minutes=3)

            send_email(email, "重置密码验证码", f"您的验证码是: {self.verification_code}")

            self.resendButton.setDisabled(True)
            self.timer.start(60000)
        except pymysql.MySQLError as e:
            self.show_tips(f"数据库错误: 发生错误: {e}")
        finally:
            cursor.close()
            connection.close()

    def enable_resend_button(self):
        self.resendButton.setDisabled(False)

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
            if 0 < self.m_Position.x() < self.resetBox_body.pos().x() + self.resetBox_body.width() and \
                    0 < self.m_Position.y() < self.resetBox_body.pos().y() + self.resetBox_body.height():
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

    def closeEvent(self, event):
        self.finished.emit()
