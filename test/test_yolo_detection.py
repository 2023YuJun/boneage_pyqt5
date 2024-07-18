from arthrosis_detection import process
from ultralytics import YOLO
import cv2
import numpy as np
from common import detection_info

# model_path = 'model/arthrosis-det.model'
# image_path = 'test_img_R.jpg'
#
# # 加载模型
# model = YOLO(model_path)


def display_cropped_images(cropped_images):
    # 获取所有图像的尺寸，假设所有图像尺寸相同
    target_size = (100, 100)  # 目标尺寸，可以根据需要调整

    # 计算每行和每列的最大图像数量
    max_per_row = 5  # 每行最多显示图像数量
    total_images = sum(len(images) for class_name, images in cropped_images.items() if class_name != "valid")
    rows = (total_images + max_per_row - 1) // max_per_row

    # 创建一个大画布，宽度是 max_per_row 个图像宽度，高度是 rows 个图像高度
    canvas_height = rows * target_size[1]
    canvas_width = max_per_row * target_size[0]
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    i = 0
    for class_name, images in cropped_images.items():
        if class_name == "valid":
            continue
        for img in images:
            resized_img = cv2.resize(img, target_size)  # 调整图像大小
            row = i // max_per_row
            col = i % max_per_row
            y1, y2 = row * target_size[1], (row + 1) * target_size[1]
            x1, x2 = col * target_size[0], (col + 1) * target_size[0]
            canvas[y1:y2, x1:x2] = resized_img
            cv2.putText(canvas, class_name, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            i += 1

    cv2.imshow("Cropped Images", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    model_path = '../model/arthrosis-det.model'
    image_path = 'images/test_img_R.jpg'

    # 加载模型
    model = YOLO(model_path)

    # 处理图像
    image = cv2.imread(image_path)
    processed_frames, left_hand, _ = process(model, [image], only_detect=False)
    for frame in processed_frames:
        cv2.imshow('Processed Image', frame)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(detection_info)

    # # 处理摄像头
    # camera_index = 0
    # cap = cv2.VideoCapture(camera_index)
    #
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         print("Failed to capture image from camera. Exiting...")
    #         break
    #     processed_frames, left_hand, _ = process(model, [frame], only_detect=False, iou=0.2, conf=0.5)
    #     for processed_frame in processed_frames:
    #         cv2.imshow('Processed Camera Frame', processed_frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #
    # cap.release()
    # cv2.destroyAllWindows()
    #
    # # 读取图像
    # image = cv2.imread(image_path)
    # if image is None:
    #     print(f"Error: Could not read image from {image_path}")
    #     return
    #
    # # 调用 process 方法
    # processed_frames, left_hand, cropped_images = process(model, [image], only_detect=False, iou=0.2, conf=0.5)
    #
    # # 显示裁剪后的图像及其类别标签
    # if cropped_images.get("valid"):
    #     display_cropped_images(cropped_images)
    # else:
    #     print("Detection was not successful.")


if __name__ == "__main__":
    main()
