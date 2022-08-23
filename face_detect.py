import face_recognition
from PIL import Image, ImageDraw
from scipy.spatial import distance
import cv2
from matplotlib import pyplot as plt
import csv


def face_locate(image):
    face_locations = face_recognition.face_locations(image)
    # for face_location in face_locations:  # 得到每一个人脸的坐标信息[上，右，下，左]
    #     top, right, bottom, left = face_location  # 解包操作，得到每张人脸的位置信息
    #     print("已识别到人脸部位，像素区域为：Top:{}, right:{}, bottom:{}, left:{}".format(top, right, bottom, left))
    # 把每一个人的头像扣出来,抠图
    # face_image = image[top:bottom, left:right] # 通过切片形式把坐标取出来
    # pil_image = Image.fromarray(face_image) #使用PIL库的fromarray方法，生成一张图片
    # pil_image.show()

    # start = (left, top)  # 左上
    # end = (right, bottom)  # 右下
    # 在图片上绘制矩形框，从start坐标开始，end坐标结束，矩形框的颜色为红色(0,255,255),矩形框粗细为2
    # cv2.rectangle(image, start, end, (0, 255, 255), thickness=2)
    return face_locations


pass


# 2.接下来得到face_landmarks_list列表
def face_landmark(image):
    face_landmarks_list = face_recognition.face_landmarks(image)
    # 对每一个人的特征点进行循环
    # for face_landmarks in face_landmarks_list:
    #     facial_features = [  # 每个人的特征点通过数组的形式列出来
    #         'chin',
    #         'left_eyebrow',
    #         'right_eyebrow',
    #         'nose_bridge',
    #         'nose_tip',
    #         'left_eye',
    #         'right_eye',
    #         'top_lip',
    #         'bottom_lip'
    #     ]
    if len(face_landmarks_list) == 0:
        return []
    return face_landmarks_list[0]


pass

# # -*- coding: utf-8 -*-
# # 自动识别人脸特征
#
# # 将jpg文件加载到numpy 数组中
# image = face_recognition.load_image_file("D:/Pycharm/PythonProject/attempt/Mouth_judge_photo/436.jpg")
#
# # 查找图像中所有面部的所有面部特征
# face_landmarks_list = face_recognition.face_landmarks(image)
# # 打印发现的脸张数
# print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))
#
# for face_landmarks in face_landmarks_list:
#
#     # 打印此图像中每个面部特征的位置
#     facial_features = [
#         'chin',
#         'left_eyebrow',
#         'right_eyebrow',
#         'nose_bridge',
#         'nose_tip',
#         'left_eye',
#         'right_eye',
#         'top_lip',
#         'bottom_lip'
#     ]
#
# for facial_feature in facial_features: print("The {} in this face has the following points: {}".format(
# facial_feature, face_landmarks[facial_feature]))
#
#     # 让我们在图像中描绘出每个人脸特征！
#     pil_image = Image.fromarray(image)
#     d = ImageDraw.Draw(pil_image)
#
#     for facial_feature in facial_features:
#         d.line(face_landmarks[facial_feature], width=5)
#
#     pil_image.show()
#     top_lip = face_landmarks_list[0]['top_lip']
#     bottom_lip = face_landmarks_list[0]['bottom_lip']
#     mar = mouth_aspect_ratio(top_lip, bottom_lip)
#     print(mar)
