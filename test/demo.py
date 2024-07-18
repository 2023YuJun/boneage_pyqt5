#导入重要的库 ，和定义函数

import os, sys
sys.path.append('/home/aistudio/work/PaddleDetection/deploy/python/')
import yaml
import ast
from functools import reduce
from PIL import Image

import matplotlib.pyplot as plt
import cv2
import numpy as np
import paddle
import paddle.fluid as fluid
from preprocess import preprocess, Resize, Normalize, Permute, PadStride


#这个是等级对应的标准分
import math
SCORE = {'girl':{
    'Radius':[10,15,22,25,40,59,91,125,138,178,192,199,203, 210],
    'Ulna':[27,31,36,50,73,95,120,157,168,176,182,189],
    'MCPFirst':[5,7,10,16,23,28,34,41,47,53,66],
    'MCPThird':[3,5,6,9,14,21,32,40,47,51],
    'MCPFifth':[4,5,7,10,15,22,33,43,47,51],
    'PIPFirst':[6,7,8,11,17,26,32,38,45,53,60,67],
    'PIPThird':[3,5,7,9,15,20,25,29,35,41,46,51],
    'PIPFifth':[4,5,7,11,18,21,25,29,34,40,45,50],
    'MIPThird':[4,5,7,10,16,21,25,29,35,43,46,51],
    'MIPFifth':[3,5,7,12,19,23,27,32,35,39,43,49],
    'DIPFirst':[5,6,8,10,20,31,38,44,45,52,67],
    'DIPThird':[3,5,7,10,16,24,30,33,36,39,49],
    'DIPFifth':[5,6,7,11,18,25,29,33,35,39,49]
},
    'boy':{
    'Radius':[8,11,15,18,31,46,76,118,135,171,188,197,201,209],
    'Ulna':[25,30,35,43,61,80,116,157,168,180,187,194],
    'MCPFirst':[4,5,8,16,22,26,34,39,45,52,66],
    'MCPThird':[3,4,5,8,13,19,30,38,44,51],
    'MCPFifth':[3,4,6,9,14,19,31,41,46,50],
    'PIPFirst':[4,5,7,11,17,23,29,36,44,52,59,66],
    'PIPThird':[3,4,5,8,14,19,23,28,34,40,45,50],
    'PIPFifth':[3,4,6,10,16,19,24,28,33,40,44,50],
    'MIPThird':[3,4,5,9,14,18,23,28,35,42,45,50],
    'MIPFifth':[3,4,6,11,17,21,26,31,36,40,43,49],
    'DIPFirst':[4,5,6,9,19,28,36,43,46,51,67],
    'DIPThird':[3,4,5,9,15,23,29,33,37,40,49],
    'DIPFifth':[3,4,6,11,17,23,29,32,36,40,49]
    }
}
def calcBoneAge(score, sex):
    #根据总分计算对应的年龄
    if sex == 'boy':
        boneAge = 2.01790023656577 + (-0.0931820870747269)*score + math.pow(score,2)*0.00334709095418796 +\
        math.pow(score,3)*(-3.32988302362153E-05) + math.pow(score,4)*(1.75712910819776E-07) +\
        math.pow(score,5)*(-5.59998691223273E-10) + math.pow(score,6)*(1.1296711294933E-12) +\
        math.pow(score,7)* (-1.45218037113138e-15) +math.pow(score,8)* (1.15333377080353e-18) +\
        math.pow(score,9)*(-5.15887481551927e-22) +math.pow(score,10)* (9.94098428102335e-26)
        return round(boneAge,2)
    elif sex == 'girl':
        boneAge = 5.81191794824917 + (-0.271546561737745)*score + \
        math.pow(score,2)*0.00526301486340724 + math.pow(score,3)*(-4.37797717401925E-05) +\
        math.pow(score,4)*(2.0858722025667E-07) +math.pow(score,5)*(-6.21879866563429E-10) + \
        math.pow(score,6)*(1.19909931745368E-12) +math.pow(score,7)* (-1.49462900826936E-15) +\
        math.pow(score,8)* (1.162435538672E-18) +math.pow(score,9)*(-5.12713017846218E-22) +\
        math.pow(score,10)* (9.78989966891478E-26)
        return round(boneAge,2)


#13个关节对应的分类模型
arthrosis ={'MCPFirst':['MCPFirst',11],
            'MCPThird':['MCP',10],
            'MCPFifth':['MCP',10],

            'DIPFirst':['DIPFirst',11],
            'DIPThird':['DIP',11],
            'DIPFifth':['DIP',11],

            'PIPFirst':['PIPFirst',12],
            'PIPThird':['PIP',12],
            'PIPFifth':['PIP',12],

            'MIPThird':['MIP',12],
            'MIPFifth':['MIP',12],

            'Radius':['Radius',14],
            'Ulna':['Ulna',12],}

class Detector(object):
    """
    检测器类，用于图像处理、预测和结果后处理
    """

    def __init__(self, config, model_dir, use_gpu=False, run_mode='fluid', threshold=0.5):
        """
        初始化检测器
        :param config: 配置对象
        :param model_dir: 模型目录
        :param use_gpu: 是否使用GPU
        :param run_mode: 运行模式
        :param threshold: 置信度阈值
        """
        self.config = config
        # 加载预测器
        self.predictor = load_predictor(
            model_dir,
            run_mode=run_mode,
            min_subgraph_size=self.config.min_subgraph_size,
            use_gpu=use_gpu
        )

    def preprocess(self, im):
        """
        图像预处理
        :param im: 输入图像
        :return: 预处理后的输入和图像信息
        """
        preprocess_ops = []
        for op_info in self.config.preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            if op_type == 'Resize':
                new_op_info['arch'] = self.config.arch
            preprocess_ops.append(eval(op_type)(**new_op_info))
        # 进行预处理操作
        im, im_info = preprocess(im, preprocess_ops)
        # 创建输入
        inputs = create_inputs(im, im_info, self.config.arch)
        return inputs, im_info

    def postprocess(self, np_boxes, np_masks, np_lmk, im_info, threshold=0.5):
        """
        结果后处理
        :param np_boxes: 检测到的框
        :param np_masks: 检测到的掩码
        :param np_lmk: 检测到的关键点
        :param im_info: 图像信息
        :param threshold: 置信度阈值
        :return: 处理后的结果
        """
        results = {}
        # 过滤低于阈值的检测框
        expect_boxes = (np_boxes[:, 1] > threshold) & (np_boxes[:, 0] > -1)
        np_boxes = np_boxes[expect_boxes, :]
        results['boxes'] = np_boxes
        return results

    def predict(self, image, threshold=0.2, warmup=0, repeats=1, run_benchmark=False):
        """
        进行预测
        :param image: 输入图像
        :param threshold: 置信度阈值
        :param warmup: 预热次数
        :param repeats: 重复次数
        :param run_benchmark: 是否运行基准测试
        :return: 预测结果
        """
        inputs, im_info = self.preprocess(image)
        np_boxes, np_masks, np_lmk = None, None, None

        # 获取输入名并赋值
        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_tensor(input_names[i])
            input_tensor.copy_from_cpu(inputs[input_names[i]])

        for i in range(repeats):
            # 执行预测
            self.predictor.zero_copy_run()
            output_names = self.predictor.get_output_names()
            boxes_tensor = self.predictor.get_output_tensor(output_names[0])
            np_boxes = boxes_tensor.copy_to_cpu()

        results = []
        if not run_benchmark:
            # 处理预测结果
            results = self.postprocess(np_boxes, np_masks, np_lmk, im_info, threshold=threshold)
        return results


def create_inputs(im, im_info, model_arch='YOLO'):
    """
    创建输入
    :param im: 输入图像
    :param im_info: 图像信息
    :param model_arch: 模型架构
    :return: 输入字典
    """
    inputs = {}
    inputs['image'] = im
    origin_shape = list(im_info['origin_shape'])
    resize_shape = list(im_info['resize_shape'])
    pad_shape = list(im_info['pad_shape']) if im_info['pad_shape'] is not None else list(im_info['resize_shape'])
    scale_x, scale_y = im_info['scale']
    im_size = np.array([origin_shape]).astype('int32')
    inputs['im_size'] = im_size
    return inputs


def load_predictor(model_dir, run_mode='fluid', batch_size=1, use_gpu=False, min_subgraph_size=3):
    """
    加载预测器
    :param model_dir: 模型目录
    :param run_mode: 运行模式
    :param batch_size: 批处理大小
    :param use_gpu: 是否使用GPU
    :param min_subgraph_size: 最小子图大小
    :return: 预测器对象
    """
    if not use_gpu and run_mode != 'fluid':
        raise ValueError(
            "Predict by TensorRT mode: {}, expect use_gpu==True, but use_gpu == {}"
            .format(run_mode, use_gpu))
    if run_mode == 'trt_int8':
        raise ValueError("TensorRT int8 mode is not supported now, "
                         "please use trt_fp32 or trt_fp16 instead.")
    precision_map = {
        'trt_int8': fluid.core.AnalysisConfig.Precision.Int8,
        'trt_fp32': fluid.core.AnalysisConfig.Precision.Float32,
        'trt_fp16': fluid.core.AnalysisConfig.Precision.Half
    }
    config = fluid.core.AnalysisConfig(
        os.path.join(model_dir, '__model__'),
        os.path.join(model_dir, '__params__'))
    if use_gpu:
        config.enable_use_gpu(100, 0)
        config.switch_ir_optim(True)
    else:
        config.disable_gpu()

    if run_mode in precision_map.keys():
        config.enable_tensorrt_engine(
            workspace_size=1 << 10,
            max_batch_size=batch_size,
            min_subgraph_size=min_subgraph_size,
            precision_mode=precision_map[run_mode],
            use_static=False,
            use_calib_mode=False)

    config.disable_glog_info()
    config.enable_memory_optim()
    config.switch_use_feed_fetch_ops(False)
    predictor = fluid.core.create_paddle_predictor(config)
    return predictor


def predict_image(detector, image_file, threshold):
    """
    预测图像
    :param detector: 检测器对象
    :param image_file: 图像文件
    :param threshold: 置信度阈值
    :return: 预测结果
    """
    results = detector.predict(image_file, threshold)
    return results


class Config():
    """
    配置类
    """

    def __init__(self, model_dir):
        """
        初始化配置
        :param model_dir: 模型目录
        """
        deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        self.arch = yml_conf['arch']
        self.preprocess_infos = yml_conf['Preprocess']
        self.use_python_inference = yml_conf['use_python_inference']
        self.min_subgraph_size = yml_conf['min_subgraph_size']
        self.labels = yml_conf['label_list']


def predictClass(im, classifer, num_classes):
    """
    预测类别
    :param im: 输入图像
    :param classifer: 分类器
    :param num_classes: 类别数
    :return: 预测结果
    """
    from paddle.vision.transforms import Compose, Resize, Normalize, Transpose
    transforms = Compose([Resize(size=(224, 224)),
                          Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], data_format='HWC'),
                          Transpose()])
    model = paddle.vision.models.resnet50(num_classes=num_classes)
    model_path = '/home/aistudio/work/out/best_' + classifer + '_net.pdparams'
    para_state_dict = paddle.load(model_path)
    model.set_dict(para_state_dict)
    model.eval()
    im = np.expand_dims(im, 2)
    infer_data = transforms(im)
    infer_data = np.expand_dims(infer_data, 0)
    infer_data = paddle.to_tensor(infer_data, dtype='float32')
    result = model(infer_data)[0]  # 实现预测功能
    result = np.argmax(result.numpy())  # 获得最大值所在的序号
    return result

# 设置性别
sex = 'girl'

# 初始化空字典，用于存储标签和分类器
label = {}
classifer = {}

# 模型目录
model_dir = '/home/aistudio/work/PaddleDetection/inference_model/yolov3_darknet_voc'
# 图像文件
image_file = '/home/aistudio/2.jpg'

# 配置字典
configDict = {}
configDict['model_dir'] = model_dir
configDict['image_file'] = image_file
configDict['threshold'] = 0.2
configDict['use_gpu'] = False
configDict['run_mode'] = 'fluid'

# 初始化配置对象
config = Config(configDict['model_dir'])

# 初始化检测器
detector = Detector(
    config, configDict['model_dir'], use_gpu=configDict['use_gpu'], run_mode=configDict['run_mode']
)

# 进行预测
results = predict_image(detector, configDict['image_file'], configDict['threshold'])

# 检查预测结果的数量
if len(results['boxes']) != 21:
    print("推理失败")

# 将预测结果分类到不同的classifer中
for box in results['boxes']:
    if int(box[0]) not in classifer:
        classifer[int(box[0])] = []
        classifer[int(box[0])].append([int(box[2]), int(box[3]), int(box[4]), int(box[5])])
    else:
        classifer[int(box[0])].append([int(box[2]), int(box[3]), int(box[4]), int(box[5])])

# 检查每个分类器中的结果数量是否正确
if len(classifer[0]) != 1 or len(classifer[1]) != 1 or len(classifer[2]) != 1:
    raise ValueError('推理失败')
if len(classifer[3]) != 4 or len(classifer[4]) != 5 or len(classifer[5]) != 4 or len(classifer[6]) != 5:
    raise ValueError('推理失败')

# True 表示左手， False 表示右手，假设默认是左手
Hand = True

# 判断左右手，通过比较第一手指掌骨和尺骨的左边界判断
if classifer[2][0][0] > classifer[1][0][0]:
    Hand = True
else:
    Hand = False

# 分别标记每个关节的位置信息
label['Radius'] = classifer[0][0]
label['Ulna'] = classifer[1][0]
label['MCPFirst'] = classifer[2][0]

# 对4个MCP按左边界排序，并标记第三和第五掌骨
MCP = sorted(classifer[3], key=lambda x: x[0], reverse=Hand)
label['MCPThird'] = MCP[1]
label['MCPFifth'] = MCP[3]

# 对5个ProximalPhalanx按左边界排序，并标记第一、第三和第五近节指骨
PIP = sorted(classifer[4], key=lambda x: x[0], reverse=Hand)
label['PIPFirst'] = PIP[0]
label['PIPThird'] = PIP[2]
label['PIPFifth'] = PIP[4]

# 对4个MiddlePhalanx按左边界排序，并标记第三和第五中节指骨
MIP = sorted(classifer[5], key=lambda x: x[0], reverse=Hand)
label['MIPThird'] = MIP[1]
label['MIPFifth'] = MIP[3]

# 对5个DistalPhalanx按左边界排序，并标记第一、第三和第五远节指骨
DIP = sorted(classifer[6], key=lambda x: x[0], reverse=Hand)
label['DIPFirst'] = DIP[0]
label['DIPThird'] = DIP[2]
label['DIPFifth'] = DIP[4]

# 读取图像
image = cv2.imread(image_file, 0)
results = {}

# 设置字体，用于在每个关节旁边写上预测的等级
font = cv2.FONT_HERSHEY_DUPLEX

# 对每个关节进行预测并绘制结果
for key, value in label.items():
    # 获取关节分类信息
    category = arthrosis[key]
    left, top, right, bottom = value
    # 从原图中截取检测框内的图像
    image_temp = image[top:bottom, left:right]
    # 进行等级预测
    result = predictClass(image_temp, category[0], category[1])
    # 绘制框框和预测的等级
    # cv2.rectangle(image, (left, top), (right, bottom), (225, 255, 255), 2)
    # cv2.putText(image, "L:{}".format(result+1), (right+3, top+40), font, 0.9, (225, 255, 255), 2)
    results[key] = result

# 计算总得分
score = 0
for key, value in results.items():
    score += SCORE[sex][key][value]

# 根据得分计算骨龄
boneAge = calcBoneAge(score, sex)

#规范报告
report = """
第一掌骨骺分级{}级，得{}分；第三掌骨骨骺分级{}级，得{}分；第五掌骨骨骺分级{}级，得{}分；
第一近节指骨骨骺分级{}级，得{}分；第三近节指骨骨骺分级{}级，得{}分；第五近节指骨骨骺分级{}级，得{}分；
第三中节指骨骨骺分级{}级，得{}分；第五中节指骨骨骺分级{}级，得{}分；
第一远节指骨骨骺分级{}级，得{}分；第三远节指骨骨骺分级{}级，得{}分；第五远节指骨骨骺分级{}级，得{}分；
尺骨分级{}级，得{}分；桡骨骨骺分级{}级，得{}分。

RUS-CHN分级计分法，受检儿CHN总得分：{}分，骨龄约为{}岁。""".format(
                                                            results['MCPFirst']+1,SCORE[sex]['MCPFirst'][results['MCPFirst']],\
                                                            results['MCPThird']+1,SCORE[sex]['MCPThird'][results['MCPThird']],\
                                                            results['MCPFifth']+1,SCORE[sex]['MCPFifth'][results['MCPFifth']],\
                                                            results['PIPFirst']+1,SCORE[sex]['PIPFirst'][results['PIPFirst']],\
                                                            results['PIPThird']+1,SCORE[sex]['PIPThird'][results['PIPThird']],\
                                                            results['PIPFifth']+1,SCORE[sex]['PIPFifth'][results['PIPFifth']],\
                                                            results['MIPThird']+1,SCORE[sex]['MIPThird'][results['MIPThird']],\
                                                            results['MIPFifth']+1,SCORE[sex]['MIPFifth'][results['MIPFifth']],\
                                                            results['DIPFirst']+1,SCORE[sex]['DIPFirst'][results['DIPFirst']],\
                                                            results['DIPThird']+1,SCORE[sex]['DIPThird'][results['DIPThird']],\
                                                            results['DIPFifth']+1,SCORE[sex]['DIPFifth'][results['DIPFifth']],\
                                                            results['Ulna']+1,SCORE[sex]['Ulna'][results['Ulna']],\
                                                            results['Radius']+1,SCORE[sex]['Radius'][results['Radius']],\
                                                            score,boneAge)
print(report)
# plt.figure(figsize=(10,10))
# plt.imshow(image,'gray')
# plt.xticks([]),plt.yticks([])
# plt.show()