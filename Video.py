import cv2
import pygame
import sys
import time


# 初始化摄像头并令其空转一定时间，否则曝光度不够
def init_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.open(0)
    for i in range(0, 50, 1):
        # flag, frame = cap.read()
        cap.read()
    return cap
    pass


# 愤怒安抚
def alert_angry():
    pygame.init()
    pygame.mixer.init()
    # screen = pygame.display.set_mode([640, 480])
    # pygame.time.delay(1000)  # 等待1秒让mixer完成初始化
    sound = pygame.mixer.Sound("D:/Pycharm/PythonProject/ModelFile/angry.wav")
    sound.play()
    time.sleep(3.1)
    pygame.mixer.music.stop()
    pass


# 疲劳警报
def alert_sleepy():
    pygame.init()
    pygame.mixer.init()
    # screen = pygame.display.set_mode([640, 480])
    # pygame.time.delay(1000)  # 等待1秒让mixer完成初始化
    sound = pygame.mixer.Sound("D:/Pycharm/PythonProject/ModelFile/sleepy.wav")
    sound.play()
    time.sleep(3.2)
    pygame.mixer.music.stop()
