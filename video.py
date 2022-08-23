import time

# import cv2
# #
# # 获取摄像头
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# # 打开摄像头
# cap.open(0)

# while cap.isOpened():
#     # 获取画面
#     flag, frame = cap.read()
#
#     ######################画面处理1##########################
#     # 灰度图
#     # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#     # frame = cv2.medianBlur(frame, 5)
#     # img_blur = cv2.GaussianBlur(frame, ksize=(21, 21),
#     #                             sigmaX=0, sigmaY=0)
#     # frame = cv2.divide(frame, img_blur, scale=255)
#
#     # 画面显示
#     cv2.imshow('mytest', frame)
#     # 设置退出按钮
#     key_pressed = cv2.waitKey(100)
#     print('单机窗口，输入按键，电脑按键为', key_pressed, '按esc键结束')
#     if key_pressed == 27:
#         break
# for i in range(0, 50, 1):
#     flag, frame = cap.read()
# if cap.isOpened:
#     flag, frame = cap.read()
#     cv2.imwrite("D:/Pycharm/PythonProject/attempt/test_cemera/image4.jpg", frame)
#     time.sleep(5)
# if cap.isOpened:
#     flag, frame = cap.read()
#     cv2.imwrite("D:/Pycharm/PythonProject/attempt/test_cemera/image5.jpg", frame)
# # 关闭摄像头
# cap.release()
# # # 关闭图像窗口
# cv2.destroyAllWindows()

from keras.models import load_model
import numpy as np
from Eye_train import eye_aspect_ratio, mouth_aspect_ratio
from face_detect import face_landmark
from sleepy_judge import sleepy_judge
import cv2
# frame = video_demo()
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
# test_file.close()
