#!usr/bin/python
import time
import cv2
import winsound
from MTCNN_DLIB import make_face_landmarks
import csv


# 需要安装opencv-python

# 读入视频文件
def read_vedio1():
    vc = cv2.VideoCapture("D:/Pycharm/PythonProject/attempt/Test_video.mp4")

    c = 1
    if vc.isOpened():
        # 判断是否正常打开
        real, frame = vc.read()
        print(real)
    else:
        real = False

    timeF = 2
    # 视频帧计数间隔频率
    number = 0
    while real:
        real, frame = vc.read()
        # print(real, frame)
        if c % timeF == 0:
            # 每隔timeF帧进行储存为图像，（注意保存地址必须全部为英文和数字，不能含有中文或者中文字符，我这个错误就是刚开始不知道是地址中文原因，找了好久。）
            cv2.imwrite("D:/Pycharm/PythonProject/attempt/Test_photo/" + str(number) + ".jpg", frame)
            number += 1
        c += 1
        cv2.waitKey(1)
    vc.release()


pass


def read_vedio2():
    training_file = open("D:/Pycharm/PythonProject/attempt/Eye_judge_dataset_test.csv", mode="w", newline="")
    csv_write = csv.writer(training_file)
    i = 0
    while i < 400:
        image = cv2.imread("D:/Pycharm/PythonProject/attempt/Eye_judge_photo/" + str(i) + ".jpg")
        landmark = make_face_landmarks(image)
        left = (max(landmark[41][0, 1], landmark[40][0, 1]) - min(landmark[37][0, 1], landmark[38][0, 1])) / (
                landmark[39][0, 0] - landmark[36][0, 0])
        right = (max(landmark[47][0, 1], landmark[46][0, 1]) - min(landmark[43][0, 1], landmark[44][0, 1])) / (
                landmark[45][0, 0] - landmark[42][0, 0])
        csv_write.writerow([left, right])
        print(i, "\n")
        i = i + 1
    pass
    training_file.close()


pass


def video_demo(cap):
    if cap.isOpened():
        ret, frame = cap.read()
    # 释放资源
    return frame


pass

# video_demo()
# cv2.destroyAllWindows()


def bell_demo():
    duration = 500  # millisecond
    freq = 700  # Hz
    for i in range(0, 3, 1):
        winsound.Beep(freq, duration)
    pass


pass

