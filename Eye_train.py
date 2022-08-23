import cv2
import csv
from face_detect import face_landmark
from scipy.spatial import distance


def write_data1():
    training_file = open("D:/Pycharm/PythonProject/attempt/Eye_judge_dataset_test.csv", mode="w", newline="")
    csv_write = csv.writer(training_file)
    i = 400
    while i < 450:
        image = cv2.imread("D:/Pycharm/PythonProject/attempt/Eye_judge_photo/" + str(i) + ".jpg")
        landmark = face_landmark(image)
        left_eye = landmark['left_eye']
        right_eye = landmark['right_eye']
        left = (max(left_eye[4][1], left_eye[5][1]) - min(left_eye[1][1], left_eye[2][1])) / (
                left_eye[3][0] - left_eye[0][0])
        right = (max(right_eye[4][1], right_eye[5][1]) - min(right_eye[1][1], right_eye[2][1])) / (
                right_eye[3][0] - right_eye[0][0])
        csv_write.writerow([left, right])
        print(i, "\n")
        i = i + 1
    pass
    training_file.close()

    pass


def eye_aspect_ratio(eye):  # 计算EAR
    a = distance.euclidean(eye[1], eye[5])
    b = distance.euclidean(eye[2], eye[4])
    c = distance.euclidean(eye[0], eye[3])
    ear = max(a, b) / c
    return ear
    pass


def mouth_aspect_ratio(top_lip, bottom_lip):  # 计算MAR
    a = distance.euclidean(top_lip[2], bottom_lip[4])
    b = distance.euclidean(top_lip[4], bottom_lip[2])
    c = distance.euclidean(bottom_lip[0], bottom_lip[6])
    mar = (a + b) / (2.0 * c)
    return mar
    pass


def eye_rectangle(eye):
    return (max(eye[1], eye[2]), eye[0]), (min(eye[5], eye[4]), eye[3])
    pass


def mouth_rectangle(top_lip, bottom_lip):
    return (max(top_lip[2], top_lip[4]), bottom_lip[0]), (min(bottom_lip[4], bottom_lip[2]), bottom_lip[6])
    pass


def write_data2():
    training_file = open("D:/Pycharm/PythonProject/attempt/Eye_judge_dataset_train2.csv", mode="w", newline="")
    csv_write = csv.writer(training_file)
    i = 0
    while i < 400:
        image = cv2.imread("D:/Pycharm/PythonProject/attempt/Eye_judge_photo/" + str(i) + ".jpg")
        landmark = face_landmark(image)
        left_eye = landmark['left_eye']
        right_eye = landmark['right_eye']
        left = eye_aspect_ratio(left_eye)
        right = eye_aspect_ratio(right_eye)
        csv_write.writerow([left, right])
        print(i, "\n")
        i = i + 1
    pass
    training_file.close()
    pass


def write_data3():
    training_file = open("D:/Pycharm/PythonProject/attempt/Mouth_judge_dataset_train.csv", mode="w", newline="")
    csv_write = csv.writer(training_file)
    i = 0
    while i < 400:
        image = cv2.imread("D:/Pycharm/PythonProject/attempt/Mouth_judge_photo/" + str(i) + ".jpg")
        landmark = face_landmark(image)
        top_lip = landmark['top_lip']
        bottom_lip = landmark['bottom_lip']
        mar = mouth_aspect_ratio(top_lip, bottom_lip)
        csv_write.writerow([mar])

        print(i, "   ", mar, "\n")
        i = i + 1
    pass
    training_file.close()
    pass
