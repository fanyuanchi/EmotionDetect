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


# 计算单眉EBAR值
def eyebrow_aspect_ratio(eyebrow):
    x1, x2, y1, y2 = eyebrow[0][0], eyebrow[0][0], eyebrow[0][1], eyebrow[0][1]
    for point in eyebrow:
        x1 = point[0] if point[0] < x1 else x1
        x2 = point[0] if point[0] > x2 else x2
        y1 = point[1] if point[1] < y1 else y1
        y2 = point[1] if point[1] > y2 else y2

    new_eyebrow_x = []
    new_eyebrow_y = []
    for point in eyebrow:
        new_x = (point[0] - x1) / (x2 - x1)
        new_y = (point[1] - y1) / (y2 - y1)
        new_eyebrow_x.append(new_x)
        new_eyebrow_y.append(new_y)
    # print(x1, x2, y1, y2, "\n")
    return new_eyebrow_x, new_eyebrow_y
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


# 制作EBAR数据集，start与end分别为图片起始与终止序号，dateset_path和result_path是数据集、结果集路径
# photo_folder_path是图片文件夹路径
def write_EBAR(start, end, dateset_path, result_path, photo_folder_path):
    dataset_file = open(dateset_path, mode="w", newline="")
    result_file = open(result_path, mode="w", newline="")
    csvdataset_write = csv.writer(dataset_file)
    csvresult_write = csv.writer(result_file)
    idx = start
    while idx < end:
        image = cv2.imread(photo_folder_path + str(idx) + ".jpg")
        landmark = face_landmark(image)

        eyebrow = []
        for left_eyebrow_point in landmark['left_eyebrow']:
            eyebrow.append(left_eyebrow_point)
        for right_euebrow_point in landmark['right_eyebrow']:
            eyebrow.append(right_euebrow_point)

        eyebrow_x, eyebrow_y = eyebrow_aspect_ratio(eyebrow)
        for x in eyebrow_x:
            csvdataset_write.writerow([x])
        for y in eyebrow_y:
            csvresult_write.writerow([y])
        # print(idx, left_ebar, right_ebar, "\n")
        idx = idx + 1
    pass
    dataset_file.close()
    pass
