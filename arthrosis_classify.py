import math

import common
from common import SCORE, classify_info, CATEGORY_MAPPING, REPORT_TEMPLATE


def classify_objects(model, image, iou=0.2, conf=0.5, stream=False, verbose=False):
    """
    使用指定模型在图像中分类对象。

    参数：
        model: 进行对象分类的模型。
        image: 输入的图像。
        iou (float): IOU阈值，默认为0.2。
        conf (float): 置信度阈值，默认为0.5。
        stream (bool): 是否启用流模式，默认为False。
        verbose (bool): 是否打印详细信息，默认为False。

    返回：
        results: 分类结果对象列表，或者None如果分类失败。
    """
    results = model(image, iou=iou, conf=conf, stream=stream, verbose=verbose)

    if results is None:
        classify_info.append("Error: Model returned None")
        return None

    # 生成检测输出信息
    info = []
    for i, result in enumerate(results):
        probs = result.probs
        if probs is None:
            info_str = f"{i}: {image.shape[1]}x{image.shape[0]} No probabilities detected"
        else:
            info_str = f"{i}: {image.shape[1]}x{image.shape[0]} "
            class_probs = {}
            for j, prob in enumerate(probs.data):  # 使用 probs.data 以获取概率值
                class_name = model.names[j]
                class_probs[class_name] = prob
            info_str += ', '.join([f"{name} {prob:.2f}" for name, prob in class_probs.items()])
            info_str += f", {result.speed['inference']:.2f}ms"
        info.append(info_str)

    # 记录速度信息
    speed_info = f"Speed: {result.speed['preprocess']:.2f}ms preprocess, {result.speed['inference']:.2f}ms inference," \
                 f" {result.speed['postprocess']:.2f}ms postprocess per image at shape {result.orig_shape}"
    info.append(speed_info)

    # 将生成的信息存储到全局变量中
    classify_info.append('\n'.join(info))

    return results


def process_images(model, cropped_images, iou=0.2, conf=0.5, stream=False, verbose=False):
    """
    处理裁剪后的图像，对每个图像进行分类。

    参数：
        model: 进行对象分类的模型。
        cropped_images: 裁剪后的图像字典。
        iou (float): IOU阈值，默认为0.2。
        conf (float): 置信度阈值，默认为0.5。
        stream (bool): 是否启用流模式，默认为False。
        verbose (bool): 是否打印详细信息，默认为False。

    返回：
        arthrosis_level: 每个类别的分类结果字典。
    """
    if not cropped_images.get("valid"):
        return {}

    arthrosis_level = {}

    for class_name, images in cropped_images.items():
        if class_name == "valid":
            continue

        img = images[0]
        results = classify_objects(model, img, iou=iou, conf=conf, stream=stream, verbose=verbose)
        if results:
            best_prob = -1
            best_level = None
            for result in results:
                if result.probs is None:
                    continue
                for j, prob in enumerate(result.probs.data):  # 使用 probs.data 以获取概率值
                    class_result = model.names[j]
                    category, level = class_result.split('_')
                    mapped_categories = CATEGORY_MAPPING.get(class_name, [class_name])

                    if category in mapped_categories and prob > best_prob:
                        best_prob = prob
                        best_level = level

            if best_level is not None:
                arthrosis_level[class_name] = best_level

    return arthrosis_level


def cal_boneage(sex, arthrosis_level):
    """
    根据性别和分类结果计算骨龄。

    参数：
        sex (str): 性别，'boy' 或 'girl'。
        arthrosis_level: 每个类别的分类结果字典。

    返回：
        boneage (float): 计算出的骨龄，或者None如果性别无效。
    """
    score = 0
    for class_name, level in arthrosis_level.items():
        level_index = int(level) - 1  # level减一对应索引
        score += SCORE[sex][class_name][level_index]

    if sex == 'boy':
        boneage = 2.01790023656577 + (-0.0931820870747269) * score + math.pow(score, 2) * 0.00334709095418796 + \
                  math.pow(score, 3) * (-3.32988302362153E-05) + math.pow(score, 4) * (1.75712910819776E-07) + \
                  math.pow(score, 5) * (-5.59998691223273E-10) + math.pow(score, 6) * (1.1296711294933E-12) + \
                  math.pow(score, 7) * (-1.45218037113138e-15) + math.pow(score, 8) * (1.15333377080353e-18) + \
                  math.pow(score, 9) * (-5.15887481551927e-22) + math.pow(score, 10) * (9.94098428102335e-26)
    elif sex == 'girl':
        boneage = 5.81191794824917 + (-0.271546561737745) * score + \
                  math.pow(score, 2) * 0.00526301486340724 + math.pow(score, 3) * (-4.37797717401925E-05) + \
                  math.pow(score, 4) * (2.0858722025667E-07) + math.pow(score, 5) * (-6.21879866563429E-10) + \
                  math.pow(score, 6) * (1.19909931745368E-12) + math.pow(score, 7) * (-1.49462900826936E-15) + \
                  math.pow(score, 8) * (1.162435538672E-18) + math.pow(score, 9) * (-5.12713017846218E-22) + \
                  math.pow(score, 10) * (9.78989966891478E-26)
    else:
        return None

    boneage = round(boneage, 2)
    common.REPORT = REPORT_TEMPLATE.format(
        arthrosis_level['MCPFirst'], SCORE[sex]['MCPFirst'][int(arthrosis_level['MCPFirst']) - 1],
        arthrosis_level['MCPThird'], SCORE[sex]['MCPThird'][int(arthrosis_level['MCPThird']) - 1],
        arthrosis_level['MCPFifth'], SCORE[sex]['MCPFifth'][int(arthrosis_level['MCPFifth']) - 1],
        arthrosis_level['PIPFirst'], SCORE[sex]['PIPFirst'][int(arthrosis_level['PIPFirst']) - 1],
        arthrosis_level['PIPThird'], SCORE[sex]['PIPThird'][int(arthrosis_level['PIPThird']) - 1],
        arthrosis_level['PIPFifth'], SCORE[sex]['PIPFifth'][int(arthrosis_level['PIPFifth']) - 1],
        arthrosis_level['MIPThird'], SCORE[sex]['MIPThird'][int(arthrosis_level['MIPThird']) - 1],
        arthrosis_level['MIPFifth'], SCORE[sex]['MIPFifth'][int(arthrosis_level['MIPFifth']) - 1],
        arthrosis_level['DIPFirst'], SCORE[sex]['DIPFirst'][int(arthrosis_level['DIPFirst']) - 1],
        arthrosis_level['DIPThird'], SCORE[sex]['DIPThird'][int(arthrosis_level['DIPThird']) - 1],
        arthrosis_level['DIPFifth'], SCORE[sex]['DIPFifth'][int(arthrosis_level['DIPFifth']) - 1],
        arthrosis_level['Ulna'], SCORE[sex]['Ulna'][int(arthrosis_level['Ulna']) - 1],
        arthrosis_level['Radius'], SCORE[sex]['Radius'][int(arthrosis_level['Radius']) - 1],
        score, boneage
    )
    return boneage
