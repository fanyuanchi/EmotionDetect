import numpy as np
from EAMcalculater import eye_aspect_ratio, mouth_aspect_ratio
from Landmarks_maker import face_landmark
from Emotion_judge import sleepy_judge
from EAMneuralwork import load_neural
from Video import init_camera
import cv2


# 主函数，实现调用摄像头获取画面，捕捉识别面部表情动作，情绪识别及警报
def main():
    idx = 0
    buffer1 = []
    buffer2 = []
    eye_model = load_neural('D:/Pycharm/PythonProject/attempt/eye_model.h5')
    mouth_model = load_neural('D:/Pycharm/PythonProject/attempt/mouth_model.h5')
    cap = init_camera()
    while True:
        flag, frame = cap.read()
        landmark = face_landmark(frame)
        cv2.imshow('mytest', frame)
        if len(landmark) == 0:
            continue
        # 眼部、嘴部数据获取
        left_eye = landmark['left_eye']
        right_eye = landmark['right_eye']
        top_lip = landmark['top_lip']
        bottom_lip = landmark['bottom_lip']
        # 眼部数据处理，使其格式可以输入神经网络
        left = eye_aspect_ratio(left_eye) * 2
        right = eye_aspect_ratio(right_eye) * 2
        eye_input = [left, right]
        eye = np.array(eye_input).reshape(1, 2)
        # 嘴部数据处理，使其格式可以输入神经网络
        mouth_input = mouth_aspect_ratio(top_lip, bottom_lip)
        mouth = np.array(mouth_input).reshape(1, 1)
        # 预测结果处理
        eye_result = eye_model.predict(eye)
        mouth_result = mouth_model.predict(mouth)
        res1 = np.argmax(eye_result)
        res2 = np.argmax(mouth_result)
        # 情绪识别
        rate = sleepy_judge(res1, res2, buffer1, buffer2, idx)
        print(idx, rate, "\n")
        key_pressed = cv2.waitKey(100)
        if key_pressed == 27:
            break
        idx = idx + 1
    pass
