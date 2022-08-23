from camera import bell_demo


def sleepy_judge(result1, result2, buffer1, buffer2, index):
    length = 20
    open_counter = 0
    close_counter = 0
    normal_counter = 0
    yawn_counter = 0
    if len(buffer1) <= length:
        buffer1.append(result1)
        buffer2.append(result2)
        return 0
    else:
        buffer1[index % length] = result1
        buffer2[index % length] = result2
    for i in range(0, length, 1):
        if buffer1[i] == 1:
            open_counter = open_counter + 1
        else:
            close_counter = close_counter + 1
        if buffer2[i] == 0 or buffer2[i] == 1:
            normal_counter = normal_counter + 1
        else:
            yawn_counter = yawn_counter + 1
    if open_counter == 0:
        rate = 0
    elif open_counter == 0 or normal_counter == 0:
        return 1
    else:
        rate = max((close_counter + yawn_counter) / (open_counter + normal_counter),
                   (close_counter/open_counter),(yawn_counter/normal_counter))
    if rate > 0.6:
        bell_demo()
        print(index, " You're sleepy!!!\n")
    return rate


pass

# print(buffer1, "\n")
# print(buffer2, "\n")
# print("NO.", index, ":", close_counter, yawn_counter, open_counter, normal_counter, rate, "\n")
