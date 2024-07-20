import cv2
from common import MAX_BOXES, DRAW_INDICES, CROP_CATEGORY_MAPPING, COLORS, detection_info


def detect_objects(model, image, iou=0.2, conf=0.5, stream=False, verbose=False):
    """
    使用指定模型在图像中检测对象。

    参数：
        model: 进行对象检测的模型。
        image: 输入的图像。
        iou (float): IOU阈值，默认为0.2。
        conf (float): 置信度阈值，默认为0.5。
        stream (bool): 是否启用流模式，默认为False。
        verbose (bool): 是否打印详细信息，默认为False。

    返回：
        results: 检测结果对象列表，或者None如果检测失败。
    """
    results = model(image, iou=iou, conf=conf, stream=stream, verbose=verbose)

    if results is None:
        detection_info.append("Error: Model returned None")
        return None

    # 生成原始检测输出信息
    info = []
    for i, result in enumerate(results):
        boxes = result.boxes
        if boxes is None:
            info_str = f"{i}: {image.shape[1]}x{image.shape[0]} No boxes detected"
        else:
            info_str = f"{i}: {image.shape[1]}x{image.shape[0]} "
            class_counts = {}
            for box in boxes:
                class_name = model.names[int(box.cls[0])]
                if class_name in class_counts:
                    class_counts[class_name] += 1
                else:
                    class_counts[class_name] = 1
            info_str += ', '.join([f"{count} {name}s" for name, count in class_counts.items()])
            info_str += f", {result.speed['inference']:.2f}ms"
        info.append(info_str)

    # 记录速度信息
    speed_info = f"Speed: {result.speed['preprocess']:.2f}ms preprocess, {result.speed['inference']:.2f}ms inference," \
                 f" {result.speed['postprocess']:.2f}ms postprocess per image at shape {result.orig_shape}"
    info.append(speed_info)

    # 将生成的信息存储到全局变量中
    detection_info.append('\n'.join(info))

    return results


def sort_and_store_boxes(results, model):
    """
    根据类名对检测框进行分类和排序。

    参数：
        results: 检测结果对象列表。
        model: 进行对象检测的模型。

    返回：
        sorted_boxes: 分类并排序后的检测框字典。
    """
    sorted_boxes = {name: [] for name in model.names.values()}
    for result in results:
        for box in result.boxes:
            class_name = model.names[int(box.cls[0])]
            sorted_boxes[class_name].append(box)
    for boxes in sorted_boxes.values():
        boxes.sort(key=lambda box: box.xyxy[0][0])
    return sorted_boxes


def determine_hand(sorted_boxes):
    """
    确定检测到的手是左手还是右手。

    参数：
        sorted_boxes: 分类并排序后的检测框字典。

    返回：
        bool: True表示左手，False表示右手。
    """
    if 'MCPFirst' in sorted_boxes and sorted_boxes['MCPFirst'] and 'Ulna' in sorted_boxes and sorted_boxes['Ulna']:
        mcpfirst_x = sorted_boxes['MCPFirst'][0].xyxy[0][0]
        ulna_x = sorted_boxes['Ulna'][0].xyxy[0][0]
        return mcpfirst_x > ulna_x
    return True


def draw_boxes(image, sorted_boxes, left_hand, restrict_draw=True):
    """
    在图像上绘制检测框。

    参数：
        image: 输入的图像。
        sorted_boxes: 分类并排序后的检测框字典。
        left_hand (bool): 是否为左手。
        restrict_draw (bool): 是否限制绘制，默认为True。

    返回：
        image: 绘制了检测框的图像。
    """
    for class_name, boxes in sorted_boxes.items():
        if not left_hand:
            boxes = boxes[::-1]  # 右手时反转检测到的边框顺序
        indices = DRAW_INDICES.get(class_name, list(range(len(boxes))))
        names = CROP_CATEGORY_MAPPING.get(class_name, [])
        for idx, box in enumerate(boxes):
            if restrict_draw and idx not in indices:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            color = COLORS[int(box.cls[0]) % len(COLORS)]
            conf = box.conf[0]
            label = f'{class_name}: {conf:.2f}' if not restrict_draw else (
                f'{names[indices.index(idx)]}: {conf:.2f}' if idx in indices and indices.index(idx) < len(
                    names) else f'{class_name}: {conf:.2f}')
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image


def check_max_boxes(sorted_boxes):
    """
    检查每个类别的检测框数量是否达到预设的最小值。

    参数：
        sorted_boxes: 分类并排序后的检测框字典。

    返回：
        bool: True表示所有类别的检测框数量都满足要求，False表示有类别未达到要求。
    """
    for class_name, max_count in MAX_BOXES.items():
        if len(sorted_boxes[class_name]) < max_count:
            print(f"Warning: {class_name} has less than {max_count} boxes!")
            return False
    return True


def crop_limited_boxes(frame, sorted_boxes, left_hand):
    """
    裁剪检测到的边框并存储裁剪后的图像。

    参数：
        frame: 输入的图像帧。
        sorted_boxes: 分类并排序后的检测框字典。
        left_hand (bool): 是否为左手。

    返回：
        cropped_images: 裁剪后的图像字典。
    """
    cropped_images = {}
    for class_name, boxes in sorted_boxes.items():
        if not left_hand:
            boxes = boxes[::-1]  # 右手时反转检测到的边框顺序
        indices = DRAW_INDICES.get(class_name, list(range(len(boxes))))
        names = CROP_CATEGORY_MAPPING.get(class_name, [])
        for idx, box in enumerate(boxes):
            if idx in indices:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cropped_img = frame[y1:y2, x1:x2].copy()  # 仅复制需要的部分
                name = names[indices.index(idx)] if idx in indices and indices.index(idx) < len(names) else class_name
                cropped_images.setdefault(name, []).append(cropped_img)
    return cropped_images


def process(model, data, iou=0.2, conf=0.5, only_detect=True, stream=False, verbose=False):
    """
    处理输入数据，进行对象检测并绘制检测框或裁剪图像。

    参数：
        model: 进行对象检测的模型。
        data: 输入的数据（图像帧列表）。
        only_detect (bool): 是否只进行检测，不进行裁剪，默认为True。
        iou (float): IOU阈值，默认为0.2。
        conf (float): 置信度阈值，默认为0.5。
        stream (bool): 是否启用流模式，默认为False。
        verbose (bool): 是否打印详细信息，默认为False。

    返回：
        processed_frames: 处理后的图像帧列表。
        left_hand (bool): 最后检测到的是否为左手。
        cropped_images: 裁剪后的图像字典。
    """
    processed_frames = []
    left_hand = True
    cropped_images = {"valid": False}

    for frame in data:
        results = detect_objects(model, frame, iou, conf, stream, verbose)
        sorted_boxes = sort_and_store_boxes(results, model)
        num_boxes = sum(len(result.boxes) for result in results)

        if not sorted_boxes or num_boxes != 21 or not check_max_boxes(sorted_boxes):  # 检查是否满足条件
            print("Warning: Detection does not meet the requirements!")
        else:
            print(
                f"21 detection boxes found and each category meets the requirements. Drawing boxes... (Left hand: {left_hand})")
        left_hand = determine_hand(sorted_boxes)
        if not only_detect:
            cropped_images = crop_limited_boxes(frame, sorted_boxes, left_hand)
            cropped_images["valid"] = True

        processed_frame = draw_boxes(frame, sorted_boxes, left_hand, restrict_draw=not only_detect)
        processed_frames.append(processed_frame)

    return processed_frames, cropped_images
