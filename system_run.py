import csv
import face_recognition
import keras
from keras.models import load_model
from scipy.spatial import distance
import numpy as np
from Eye_train import eye_aspect_ratio, mouth_aspect_ratio, eye_rectangle, mouth_rectangle
from camera import video_demo
from face_detect import face_landmark
from sleepy_judge import sleepy_judge
import cv2
import winsound
import time

idx = 0
buffer1 = []
buffer2 = []
eye_model = load_model('D:/Pycharm/PythonProject/attempt/eye_model.h5')
mouth_model = load_model('D:/Pycharm/PythonProject/attempt/mouth_model.h5')
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.open(0)
for i in range(0, 50, 1):
    flag, frame = cap.read()
while True:
    flag, frame = cap.read()
    landmark = face_landmark(frame)
    cv2.imshow('mytest', frame)
    if len(landmark) == 0:
        continue
    left_eye = landmark['left_eye']
    right_eye = landmark['right_eye']
    top_lip = landmark['top_lip']
    bottom_lip = landmark['bottom_lip']
    left = eye_aspect_ratio(left_eye) * 2
    right = eye_aspect_ratio(right_eye) * 2
    mouth_input = mouth_aspect_ratio(top_lip, bottom_lip)
    eye_input = [left, right]
    eye = np.array(eye_input).reshape(1, 2)
    mouth = np.array(mouth_input).reshape(1, 1)
    result1 = eye_model.predict(eye)
    result2 = mouth_model.predict(mouth)
    res1 = np.argmax(result1)
    res2 = np.argmax(result2)
    rate = sleepy_judge(res1, res2, buffer1, buffer2, idx)
    print(idx, rate, "\n")
    key_pressed = cv2.waitKey(100)
    if key_pressed == 27:
        break
    idx = idx + 1
pass
