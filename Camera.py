import cv2


# 读入视频文件并按某一帧率截图保存
def read_video(video_path, photo_folder_path):
    vc = cv2.VideoCapture(video_path)

    if vc.isOpened():
        # 判断是否正常打开
        real, frame = vc.read()
        print(real)
    else:
        real = False

    real_time = 1
    # 记录视频当前帧值
    timeF = 2
    # 视频帧计数间隔频率
    number = 0
    # 初始化截图序号
    while real:
        real, frame = vc.read()
        # print(real, frame)
        if real_time % timeF == 0:
            # 每隔timeF帧进行储存为图像，（注意保存地址必须全部为英文和数字，不能含有中文或者中文字符）
            cv2.imwrite(photo_folder_path + str(number) + ".jpg", frame)
            number += 1
        real_time += 1
        cv2.waitKey(1)
    vc.release()
    pass
