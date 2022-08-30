from Video import alert


# 疲劳检测函数，eye_result为眼部动作识别结果，mouth_result为嘴部动作识别结果
# eye_buffer为眼部动作记录缓冲， mouth_buffer为嘴部动作记录缓冲，index为当前结果序号
def sleepy_judge(eye_result, mouth_result, eye_buffer, mouth_buffer, index):
    length = 20
    # 眼部睁闭计数器
    open_counter = 0
    close_counter = 0
    # 嘴部正常（包括闭嘴和说话）、打呵欠计数器
    normal_counter = 0
    yawn_counter = 0
    # 不对刚刚开机时获取的数据进行分析吗，但将其记录在缓冲中
    if len(eye_buffer) <= length:
        eye_buffer.append(eye_result)
        mouth_buffer.append(mouth_result)
        return 0
    else:
        eye_buffer[index % length] = eye_result
        mouth_buffer[index % length] = mouth_result
    # 便历缓冲，计算比例rate
    for i in range(0, length, 1):
        if eye_buffer[i] == 1:
            open_counter = open_counter + 1
        else:
            close_counter = close_counter + 1
        if mouth_buffer[i] == 0 or mouth_buffer[i] == 1:
            normal_counter = normal_counter + 1
        else:
            yawn_counter = yawn_counter + 1

    if open_counter == 0 or normal_counter == 0:
        rate = 1
    else:
        rate = max((close_counter + yawn_counter) / (open_counter + normal_counter),
                   (close_counter/open_counter), (yawn_counter/normal_counter))
    if rate > 0.6:
        alert()
        print(index, " You're sleepy!!!\n")
    return rate
    pass
