from ultralytics import YOLO
from PIL import Image
import cv2

# model = YOLO(r"E:\boneage_pyqt5\model\arthrosis-det.model")
#
# # from PIL
# im1 = Image.open(r"E:\boneage_pyqt5\test_img_L.jpg")
# results = model.predict(source=im1, save=True)  # save plotted images


# 加载官方模型
model = YOLO(r"E:\boneage_pyqt5\model\arthrosis-cls.model")


# 使用模型进行预测
results = model(r"E:\boneage_pyqt5\test\temp_cropped_image.jpg")

# # 输出分类结果
# for result in results:
#     print(result)

first_result = results[0]
print(first_result)