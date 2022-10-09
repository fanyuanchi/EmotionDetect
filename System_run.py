import numpy as np
from EAMcalculater import eye_aspect_ratio, mouth_aspect_ratio, eyebrow_aspect_ratio
from Landmarks_maker import face_landmark
from Emotion_judge import sleepy_judge, angry_judge
from EAMneuralwork import load_neural
from Video import init_camera
import cv2


# 主函数，实现调用摄像头获取画面，捕捉识别面部表情动作，情绪识别及警报
def main():
    idx = 0
    buffer1 = []
    buffer2 = []
    buffer3 = []
    eye_model = load_neural('D:/Pycharm/PythonProject/ModelFile/eye_model.h5')
    mouth_model = load_neural('D:/Pycharm/PythonProject/ModelFile/mouth_model.h5')
    eyebrow_normal_model = load_neural('D:/Pycharm/PythonProject/ModelFile/eyebrow_normal_model.h5')
    eyebrow_angry_model = load_neural('D:/Pycharm/PythonProject/ModelFile/eyebrow_angry_model.h5')
    cap = init_camera()
    while True:
        flag, frame = cap.read()
        landmark = face_landmark(frame)
        cv2.imshow('mytest', frame)
        if len(landmark) == 0:
            continue
        # 眼部、嘴部、眉毛数据获取
        left_eye = landmark['left_eye']
        right_eye = landmark['right_eye']

        top_lip = landmark['top_lip']
        bottom_lip = landmark['bottom_lip']
        eyebrow = []
        for left_eyebrow_point in landmark['left_eyebrow']:
            eyebrow.append(left_eyebrow_point)
        for right_eyebrow_point in landmark['right_eyebrow']:
            eyebrow.append(right_eyebrow_point)
        eyebrow_x, eyebrow_y = eyebrow_aspect_ratio(eyebrow)
        # 眼部数据处理，使其格式可以输入神经网络
        left = eye_aspect_ratio(left_eye) * 2
        right = eye_aspect_ratio(right_eye) * 2
        eye_input = [left, right]
        eye = np.array(eye_input).reshape(1, 2)
        # 嘴部数据处理，使其格式可以输入神经网络
        mouth_input = mouth_aspect_ratio(top_lip, bottom_lip)
        mouth = np.array(mouth_input).reshape(1, 1)
        # 眉毛数据初始化，使其格式可以输入神经网络
        predictions_normal = []
        predictions_angry = []
        # eyebrow_x = eyebrow_x.astype('float32')
        # eyebrow_y = eyebrow_y.astype('float32')
        # 眼、嘴部预测结果处理
        eye_result = eye_model.predict(eye)
        mouth_result = mouth_model.predict(mouth)
        res1 = np.argmax(eye_result)
        res2 = np.argmax(mouth_result)
        # 眉毛预测结果处理
        eyebrow_result = []
        for i in range(0, len(eyebrow_x)):
            prediction_normal = eyebrow_normal_model.predict([eyebrow_x[i]])
            prediction_angry = eyebrow_angry_model.predict([eyebrow_x[i]])
            predictions_normal.append(prediction_normal)
            predictions_angry.append(prediction_angry)
        for i in range(0, len(eyebrow_x)):
            point_result = 0 if abs(predictions_normal[i] - eyebrow_y[i]) < abs(
                predictions_angry[i] - eyebrow_y[i]) else 1
            eyebrow_result.append(point_result)
        eyebrow_result.pop(9)
        if eyebrow_result[4] == 1 and eyebrow_result[5] == 1:
            extra_val = 2
        elif eyebrow_result[4] == 0 and eyebrow_result[5] == 0:
            extra_val = 0
        else:
            extra_val = 1
        res3 = 1 if eyebrow_result.count(1) + extra_val > 4 else 0
        # 情绪短路判断
        if res3 == 1:
            res1 = 1
        # 情绪识别
        rate2, judge2 = angry_judge(res3, res2, buffer3, buffer2, idx)
        rate1, judge1 = sleepy_judge(res1, res2, buffer1, buffer2, idx)
        # print(res1, res2, res3, "\n")
        print(idx, rate1, rate2, "\n")
        key_pressed = cv2.waitKey(100)
        if key_pressed == 27:
            break
        idx = idx + 1
    pass
