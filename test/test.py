import numpy as np
import cv2
from PIL import Image
import sys
import os
ROOT = os.path.dirname(os.path.abspath(__file__)) + "/../"
sys.path.append(ROOT)
from easy_ViTPose import VitInference

img = Image.open('/home/nhan/Desktop/DATN/repo/Action_Recogntion/my-own/assets/Rose.jpg')
img = np.array(img)

model = VitInference(model='models/PoseEstimation/vitpose-25-l.onnx',yolo='models/PoseEstimation/yolov5s.onnx',model_name='l',yolo_size=320, is_video= False)
frame_keypoints  =  model.inference(img)
img = model.draw(show_yolo=True)

cv2.imshow(img[..., ::-1])
