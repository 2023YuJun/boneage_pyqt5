from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO(r"E:\boneage_pyqt5\pt\yolov8n.pt")

# from PIL
im1 = Image.open(r"E:\boneage_pyqt5\test_img.jpg")
results = model.predict(source=im1, save=True)  # save plotted images
