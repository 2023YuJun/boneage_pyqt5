# from ultralytics import YOLO
# import cv2
# from arthrosis_classify import process_images, cal_boneage
# from arthrosis_detection import process
# import common
#
# model_path_1 = '../model/atrs-n-det.pt'
# model_path_2 = '../model/atrs-n-cls.pt'
# image_path = 'images/PIP_1.png'
# # image_path = 'PIP_1.png'
#
# # 加载模型
# model1 = YOLO(model_path_1)
# model2 = YOLO(model_path_2)
#
# image = cv2.imread(image_path)
# # processed_frames, left_hand, cropped_images = process(model1, [image], only_detect=False)
# classifications = process_images(model2, image, iou=0.1, conf=0.1)
# cv2.imshow(classifications)
# # print(cal_boneage(sex="girl", arthrosis_level=classifications))
# # print(common.REPORT)

import cv2
from ultralytics import YOLO
from arthrosis_classify import process_images


def load_image(image_path):
    """
    加载图像文件。

    参数：
        image_path (str): 图像文件的路径。

    返回：
        image: 加载的图像。
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
    return image


def load_video(video_path):
    """
    加载视频文件。

    参数：
        video_path (str): 视频文件的路径。

    返回：
        video: 视频捕获对象。
    """
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not load video from {video_path}")
    return video


def process_and_display(model, source):
    """
    处理视频或图像，并显示标注结果。

    参数：
        model: YOLO 模型实例。
        source: 图像文件路径或视频捕获对象。
    """
    if isinstance(source, str):
        # 处理图像文件
        image = load_image(source)
        if image is not None:
            labeled_images = process_images(model, [image])
            if labeled_images:
                for labeled_image in labeled_images:
                    cv2.imshow('Labeled Image', labeled_image)
                    cv2.waitKey(0)
                cv2.destroyAllWindows()
    elif isinstance(source, cv2.VideoCapture):
        # 处理视频文件
        while True:
            ret, frame = source.read()
            if not ret:
                break
            labeled_images = process_images(model, [frame])
            if labeled_images:
                for labeled_image in labeled_images:
                    cv2.imshow('Labeled Video', labeled_image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        source.release()
        cv2.destroyAllWindows()
    else:
        print("Error: Unsupported source type. Please provide a valid image path or video capture object.")


if __name__ == "__main__":
    # 加载模型，并指定使用GPU
    model = YOLO('../model/atrs-n-cls.pt')

    # 输入源（图像路径或视频路径）
    input_source = "images/PIP_1.png"

    if input_source.lower().endswith(('.png', '.jpg', '.jpeg')):
        process_and_display(model, input_source)
    else:
        video = load_video(input_source)
        if video:
            process_and_display(model, video)
