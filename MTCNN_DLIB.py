import cv2
import numpy as np
import os
import dlib
import logging
from matplotlib import pyplot as plt
from mtcnn import MTCNN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
logging.disable(30)

detector = MTCNN()
predictor = dlib.shape_predictor("D:/Pycharm/PythonProject/NeuralWork/shape_predictor_68_face_landmarks.dat")


#
def detect_face(img):
    shapes = np.array(img).shape

    face_arr = dlib.rectangles()
    # 得到高度和宽度 做异常处理
    height = shapes[0]
    weight = shapes[1]
    # 检测人脸
    detect_result = detector.detect_faces(img)
    # 如果检测不到人脸、 返回空
    if len(detect_result) == 0:
        return []
    else:
        for item in detect_result:
            box = item['box']
            # 因为预测是 给出左上角的点坐标【0，1】  以及 长宽【2，3】  所以需要转换
            top = box[1]
            bottom = box[1] + box[3]
            left = box[0]
            right = box[0] + box[2]
            # 因为左上角的点可能会在图片范围外 所以要异常处理
            if top < 0:
                top = 0
            if left < 0:
                left = 0
            if bottom > height:
                bottom = height
            if right > weight:
                right = weight

            rectangle = dlib.rectangle(left, top, right, bottom)
            # 创建一个rectangles空对象
            face_arr.append(rectangle)
    return face_arr


def make_landmarks(face_arr, img):
    if not len(face_arr) == 0:
        for i in range(0, len(face_arr)):
            landmarks = np.matrix([[p.x, p.y] for p in predictor(img, face_arr[i]).parts()])
            for idx, point in enumerate(landmarks):
                # 68点的坐标
                pos = (point[0, 0], point[0, 1])
                print(idx, pos)

                # 利用cv2.circle给每个特征点画一个圈，共68个
                cv2.circle(img, pos, 2, color=(0, 255, 0), thickness=-1)
                # 利用cv2.putText输出1-68
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, str(idx + 1), pos, font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.show()
        return landmarks

    else:
        print("No face is detected")
        return None

#
# image = cv2.imread("D:/Pycharm/PythonProject/attempt/Eye_judge_photo/23.jpg")
# # 图像锐化
# # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 定义一个核
# # image = cv2.filter2D(image, -1, kernel=kernel)
# # 图像增亮
# # image = cv2.add(image, image)
# face_array = detect_face(image)
# landmark = make_landmarks(face_array, image)

#######################################################
def make_face_landmarks(img):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("D:/Pycharm/PythonProject/NeuralWork/shape_predictor_68_face_landmarks.dat")
    # 取灰度
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 人脸数rects
    rects = detector(img_gray, 0)
    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rects[i]).parts()])
    return landmarks


pass

# for idx, point in enumerate(landmarks):
#     # 68点的坐标
#     pos = (point[0, 0], point[0, 1])
#     print(idx, pos)
#
#     # 利用cv2.circle给每个特征点画一个圈，共68个
#     cv2.circle(img, pos, 3, color=(0, 255, 0), thickness=-1)
#     # 利用cv2.putText输出1-68
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     cv2.putText(img, str(idx + 1), pos, font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
