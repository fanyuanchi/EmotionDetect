# from EAMneuralwork import train_EBAR, print_history2
#
# history, model = train_EBAR("D:/Pycharm/PythonProject/attempt/eyebrow_normal_train.csv",
#                             "D:/Pycharm/PythonProject/attempt/eyebrow_normal_trainresult.csv",
#                             "D:/Pycharm/PythonProject/attempt/eyebrow_normal_test.csv",
#                             "D:/Pycharm/PythonProject/attempt/eyebrow_normal_testresult.csv")
# print_history2(history)

# from EAMcalculater import eyebrow_aspect_ratio
# import cv2
# from Landmarks_maker import draw_landmark
# from PIL import Image, ImageDraw

# image = cv2.imread("D:/Pycharm/PythonProject/attempt/Eyebrow_normal_photo/21.jpg")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# landmarks = draw_landmark(image)
# eyebrow = []
#
# for left_eyebrow_point in landmarks['left_eyebrow']:
#     eyebrow.append(left_eyebrow_point)
# for right_euebrow_point in landmarks['right_eyebrow']:
#     eyebrow.append(right_euebrow_point)
# print(eyebrow, "\n")
#
# eyebrow = eyebrow_aspect_ratio(eyebrow)
# print(eyebrow)


# from EAMneuralwork import load_neural
# import cv2
# from Landmarks_maker import face_landmark
# from EAMcalculater import eyebrow_aspect_ratio
# import numpy as np
# import matplotlib.pyplot as plt
#
# normal = load_neural('D:/Pycharm/PythonProject/attempt/eyebrow_normal_model.h5')
# angry = load_neural('D:/Pycharm/PythonProject/attempt/eyebrow_angry_model.h5')
# image = cv2.imread("D:/Pycharm/PythonProject/attempt/Eye_judge_photo/139.jpg")
# landmark = face_landmark(image)
# eyebrow = []
# for left_eyebrow_point in landmark['left_eyebrow']:
#     eyebrow.append(left_eyebrow_point)
# for right_eyebrow_point in landmark['right_eyebrow']:
#     eyebrow.append(right_eyebrow_point)
# eyebrow_x, eyebrow_y = eyebrow_aspect_ratio(eyebrow)
# normal_y = []
# angry_y = []
# for x in eyebrow_x:
#     x = np.array(x).reshape(1, 1)
#     normal_y.append(normal.predict(x))
#     angry_y.append(angry.predict(x))
#
# normal_y = np.array(normal_y).reshape(10, 1)
# angry_y = np.array(angry_y).reshape(10, 1)
# eyebrow_x = np.array(eyebrow_x).reshape(10, 1)
# eyebrow_y = np.array(eyebrow_y).reshape(10, 1)
# print(eyebrow_x, "\n")
# print(eyebrow_y, "\n")
#
# print(normal_y)
# print(angry_y)
#
# eyebrow_x = list(np.array(eyebrow_x).flatten())
# eyebrow_y = list(np.array(eyebrow_y).flatten())
#
# normal_y = list(np.array(normal_y).flatten())
# angry_y = list(np.array(angry_y).flatten())
#
# fig = plt.figure()  # 创建画布
# ax = fig.add_subplot(111)
#
# p1 = ax.scatter(eyebrow_x, eyebrow_y, marker='.', color='black', s=15)
# p2 = ax.scatter(eyebrow_x, normal_y, marker='.', color='red', s=15)
# p3 = ax.scatter(eyebrow_x, angry_y, marker='.', color='blue', s=15)
# plt.show()  # 显示散点图


# from EAMneuralwork import load_neural, load_data, print_history3
# import csv
# import numpy as np
#
# (x_train, y_train), (x_test, y_test) = load_data("D:/Pycharm/PythonProject/attempt/eyebrow_angry_train.csv",
#                                                  "D:/Pycharm/PythonProject/attempt/eyebrow_angry_trainresult.csv",
#                                                  "D:/Pycharm/PythonProject/attempt/eyebrow_angry_test.csv",
#                                                  "D:/Pycharm/PythonProject/attempt/eyebrow_angry_testresult.csv")
# normal = load_neural('D:/Pycharm/PythonProject/attempt/eyebrow_normal_model.h5')
# angry = load_neural('D:/Pycharm/PythonProject/attempt/eyebrow_angry_model.h5')
#
# print_history3(x_train, y_train, normal, angry)

from System_run import main


main()

