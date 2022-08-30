import cv2
import csv
from Landmarks_maker import face_landmark
from scipy.spatial import distance


# 计算单眼EAR值
def eye_aspect_ratio(eye):
    a = distance.euclidean(eye[1], eye[5])
    b = distance.euclidean(eye[2], eye[4])
    c = distance.euclidean(eye[0], eye[3])
    ear = max(a, b) / c
    # a, b为单眼纵距，c为单眼横距
    return ear
    pass


# 计算嘴部MAR值，注意EAR算法与MAR算法不一致
# 由于单眼并非对称的纺锤形，而嘴部多为对称结构，所以在计算EAR时采用最大纵距分离眼部动作特征，而计算MAR时采用平均纵距模糊嘴部动作
def mouth_aspect_ratio(top_lip, bottom_lip):
    a = distance.euclidean(top_lip[2], bottom_lip[4])
    b = distance.euclidean(top_lip[4], bottom_lip[2])
    c = distance.euclidean(bottom_lip[0], bottom_lip[6])
    mar = (a + b) / (2.0 * c)
    # a, b为嘴部纵距，c为嘴部横距
    return mar
    pass


# 制作EAR数据集，start与end是视频截图序号，dateset_path是数据集文件路径，photo_folder_path是图片文件夹路径
def write_EAR(start, end, dateset_path, photo_folder_path):
    training_file = open(dateset_path, mode="w", newline="")
    csv_write = csv.writer(training_file)
    idx = start
    while idx < end:
        image = cv2.imread(photo_folder_path + str(idx) + ".jpg")
        landmark = face_landmark(image)

        left_eye = landmark['left_eye']
        right_eye = landmark['right_eye']

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        csv_write.writerow([left_ear, right_ear])
        # print(idx, left_ear, right_ear, "\n")
        idx = idx + 1
    pass
    training_file.close()
    pass


# 制作MAR数据集，函数各参数意义与write_EAR(start, end, dateset_path, photo_folder_path)一样
def write_MAR(start, end, dateset_path, photo_folder_path):
    training_file = open(dateset_path, mode="w", newline="")
    csv_write = csv.writer(training_file)
    idx = start
    while idx < end:
        image = cv2.imread(photo_folder_path + str(idx) + ".jpg")
        landmark = face_landmark(image)

        top_lip = landmark['top_lip']
        bottom_lip = landmark['bottom_lip']

        mar = mouth_aspect_ratio(top_lip, bottom_lip)

        csv_write.writerow([mar])
        # print(idx, mar, "\n")
        idx = idx + 1
    pass
    training_file.close()
    pass
