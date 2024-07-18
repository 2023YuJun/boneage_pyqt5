from ultralytics import YOLO
import cv2
from arthrosis_classify import process_images, cal_boneage
from arthrosis_detection import process
import common

model_path_1 = '../model/arthrosis-det.model'
model_path_2 = '../model/arthrosis-cls.model'
image_path = 'images/test_img_L.jpg'
# image_path = 'PIP_1.png'

# 加载模型
model1 = YOLO(model_path_1)
model2 = YOLO(model_path_2)

image = cv2.imread(image_path)
processed_frames, left_hand, cropped_images = process(model1, [image], only_detect=False)
classifications = process_images(model2, cropped_images, iou=0.1, conf=0.1)
print(cal_boneage(sex="girl", arthrosis_level=classifications))
print(common.REPORT)
