import cv2
import winsound


# 初始化摄像头并令其空转一定时间，否则曝光度不够
def init_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.open(0)
    for i in range(0, 50, 1):
        # flag, frame = cap.read()
        cap.read()
    return cap
    pass


# 发出警报
def alert():
    duration = 500  # 单次警报时长
    freq = 700  # 单次警报音频
    for i in range(0, 3, 1): # 重复警报三次
        winsound.Beep(freq, duration)
    pass
