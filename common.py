import json
import random
import string

import bcrypt
import pymysql
import yagmail

# 全局变量用于存储检测输出信息
detection_info = []

# 全局变量用于存储分类输出信息
classify_info = []

# 最终报告
REPORT = ""

# 定义类别
CLASSIFY = ["DIPFirst", "DIP", "MIP", "PIPFirst", "PIP", "MCPFirst", "MCP", "Radius", "Ulna"]
DETECT = ["DistalPhalanx", "MiddlePhalanx", "ProximalPhalanx", "MCPFirst", "MCP", "Radius", "Ulna"]

# 定义类别映射
CATEGORY_MAPPING = {
    "MCPFirst": ["MCPFirst"],
    "MCPThird": ["MCP"],
    "MCPFifth": ["MCP"],
    "PIPFirst": ["PIPFirst"],
    "PIPThird": ["PIP"],
    "PIPFifth": ["PIP"],
    "MIPThird": ["MIP"],
    "MIPFifth": ["MIP"],
    "DIPFirst": ["DIPFirst"],
    "DIPThird": ["DIP"],
    "DIPFifth": ["DIP"],
    "Radius": ["Radius"],
    "Ulna": ["Ulna"]
}

# 定义一个字典来控制每个类别的最大绘制数量
MAX_BOXES = {
    'DistalPhalanx': 5,
    'MCP': 4,
    'MCPFirst': 1,
    'MiddlePhalanx': 4,
    'ProximalPhalanx': 5,
    'Radius': 1,
    'Ulna': 1
}

# 定义一个字典来控制需要绘制的边框下标
DRAW_INDICES = {
    'DistalPhalanx': [0, 2, 4],
    'MCP': [0, 2],
    'MiddlePhalanx': [0, 2],
    'ProximalPhalanx': [0, 2, 4]
}

# 定义类别映射到具体名称的字典
CROP_CATEGORY_MAPPING = {
    'DistalPhalanx': ['DIPFifth', 'DIPThird', 'DIPFirst'],
    'MCP': ['MCPFifth', 'MCPThird'],
    'MiddlePhalanx': ['MIPFifth', 'MIPThird'],
    'ProximalPhalanx': ['PIPFifth', 'PIPThird', 'PIPFirst']
}

# 定义颜色列表，每个类别使用不同的颜色
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (139, 0, 0), (0, 100, 0),
          (0, 0, 139)]

# 等级对应的标准分
SCORE = {
    'girl': {
        'Radius': [10, 15, 22, 25, 40, 59, 91, 125, 138, 178, 192, 199, 203, 210],
        'Ulna': [27, 31, 36, 50, 73, 95, 120, 157, 168, 176, 182, 189],
        'MCPFirst': [5, 7, 10, 16, 23, 28, 34, 41, 47, 53, 66],
        'MCPThird': [3, 5, 6, 9, 14, 21, 32, 40, 47, 51],
        'MCPFifth': [4, 5, 7, 10, 15, 22, 33, 43, 47, 51],
        'PIPFirst': [6, 7, 8, 11, 17, 26, 32, 38, 45, 53, 60, 67],
        'PIPThird': [3, 5, 7, 9, 15, 20, 25, 29, 35, 41, 46, 51],
        'PIPFifth': [4, 5, 7, 11, 18, 21, 25, 29, 34, 40, 45, 50],
        'MIPThird': [4, 5, 7, 10, 16, 21, 25, 29, 35, 43, 46, 51],
        'MIPFifth': [3, 5, 7, 12, 19, 23, 27, 32, 35, 39, 43, 49],
        'DIPFirst': [5, 6, 8, 10, 20, 31, 38, 44, 45, 52, 67],
        'DIPThird': [3, 5, 7, 10, 16, 24, 30, 33, 36, 39, 49],
        'DIPFifth': [5, 6, 7, 11, 18, 25, 29, 33, 35, 39, 49]
    },
    'boy': {
        'Radius': [8, 11, 15, 18, 31, 46, 76, 118, 135, 171, 188, 197, 201, 209],
        'Ulna': [25, 30, 35, 43, 61, 80, 116, 157, 168, 180, 187, 194],
        'MCPFirst': [4, 5, 8, 16, 22, 26, 34, 39, 45, 52, 66],
        'MCPThird': [3, 4, 5, 8, 13, 19, 30, 38, 44, 51],
        'MCPFifth': [3, 4, 6, 9, 14, 19, 31, 41, 46, 50],
        'PIPFirst': [4, 5, 7, 11, 17, 23, 29, 36, 44, 52, 59, 66],
        'PIPThird': [3, 4, 5, 8, 14, 19, 23, 28, 34, 40, 45, 50],
        'PIPFifth': [3, 4, 6, 10, 16, 19, 24, 28, 33, 40, 44, 50],
        'MIPThird': [3, 4, 5, 9, 14, 18, 23, 28, 35, 42, 45, 50],
        'MIPFifth': [3, 4, 6, 11, 17, 21, 26, 31, 36, 40, 43, 49],
        'DIPFirst': [4, 5, 6, 9, 19, 28, 36, 43, 46, 51, 67],
        'DIPThird': [3, 4, 5, 9, 15, 23, 29, 33, 37, 40, 49],
        'DIPFifth': [3, 4, 6, 11, 17, 23, 29, 32, 36, 40, 49]
    }
}

# 报告模板
REPORT_TEMPLATE = """第一掌骨骺分级{}级，得{}分；第三掌骨骨骺分级{}级，得{}分；第五掌骨骨骺分级{}级，得{}分；
第一近节指骨骨骺分级{}级，得{}分；第三近节指骨骨骺分级{}级，得{}分；第五近节指骨骨骺分级{}级，得{}分；
第三中节指骨骨骺分级{}级，得{}分；第五中节指骨骨骺分级{}级，得{}分；
第一远节指骨骨骺分级{}级，得{}分；第三远节指骨骨骺分级{}级，得{}分；第五远节指骨骨骺分级{}级，得{}分；
尺骨分级{}级，得{}分；桡骨骨骺分级{}级，得{}分。

RUS-CHN分级计分法，受检儿CHN总得分：{}分，骨龄约为{}岁。
"""


# 读取数据库配置
def load_db_config():
    with open('config/db_config.json', 'r') as file:
        return json.load(file)


db_config = load_db_config()


# 连接数据库
def connect_db():
    connection = pymysql.connect(
        host=db_config['host'],
        user=db_config['user'],
        password=db_config['password'],
        database=db_config['database']
    )
    return connection


# 通用哈希方法
def hash_data(data):
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(data.encode('utf-8'), salt)
    return hashed.decode('utf-8')


# 通用验证方法
def verify_data(stored_data, provided_data):
    return bcrypt.checkpw(provided_data.encode('utf-8'), stored_data.encode('utf-8'))


# 生成验证码
def generate_verification_code(length=6):
    return ''.join(random.choice(string.digits) for _ in range(length))


# 发送邮件
def send_email(to_email, subject, content):
    connection = connect_db()
    cursor = connection.cursor()

    try:
        cursor.execute("SELECT email, password, host, port FROM email_accounts LIMIT 1")
        account = cursor.fetchone()

        if not account:
            raise Exception("未找到邮箱账号信息")

        email, password, host, port = account
        yag = yagmail.SMTP(user=email, password=password, host=host, port=port)
        yag.send(to=to_email, subject=subject, contents=content)

    except pymysql.MySQLError as e:
        print(f"数据库错误: 发生错误: {e}")

    except Exception as e:
        print(f"发送邮件失败: {e}")

    finally:
        cursor.close()
        connection.close()


# 获取服务器时间
def get_server_time():
    connection = connect_db()
    cursor = connection.cursor()

    try:
        cursor.execute("SELECT NOW()")
        result = cursor.fetchone()
        return result[0]  # 返回服务器当前时间
    except pymysql.MySQLError as e:
        print(f"数据库错误: {e}")
    finally:
        cursor.close()
        connection.close()


# 检查密码安全性
def is_password_strong(password):
    if len(password) < 8:
        return False
    if sum([bool(c.isupper()) for c in password]) + sum([bool(c.islower()) for c in password]) + \
            sum([bool(c.isdigit()) for c in password]) + sum([bool(c in string.punctuation) for c in password]) < 3:
        return False
    return True
