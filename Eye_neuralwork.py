import imageio
import cv2
import keras
from keras.models import load_model
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adadelta
import matplotlib.pyplot as plt
from Eye_train import eye_aspect_ratio
from face_detect import face_landmark


def read_data():
    train_data_file = open("D:/Pycharm/PythonProject/attempt/Mouth_judge_dataset_train.csv", "r")
    train_resu_file = open("D:/Pycharm/PythonProject/attempt/Mouth_judge_dataset_trainresult.csv", "r")
    test_data_file = open("D:/Pycharm/PythonProject/attempt/Mouth_judge_dataset_test.csv", "r")
    test_resu_file = open("D:/Pycharm/PythonProject/attempt/Mouth_judge_dataset_testresult.csv", "r")
    train_data_list = train_data_file.readlines()
    train_sesu_list = train_resu_file.readlines()
    test_data_list = test_data_file.readlines()
    test_resu_list = test_resu_file.readlines()
    train_data_file.close()
    train_resu_file.close()
    test_data_file.close()
    test_resu_file.close()
    train_inputs = []
    train_result = []
    test_inputs = []
    test_result = []
    for record in train_data_list:
        all_values = record.split(',')
        train_inputs.append(all_values[:])
    pass
    for record in train_sesu_list:
        all_values = record.split(',')
        train_result.append(all_values[:])
    pass
    for record in test_data_list:
        all_values = record.split(',')
        test_inputs.append(all_values[:])
    pass
    for record in test_resu_list:
        all_values = record.split(',')
        test_result.append(all_values[:])
    pass
    train_input = np.array(train_inputs)
    train_resul = np.array(train_result)
    test_input = np.array(test_inputs)
    test_resul = np.array(test_result)
    return (train_input, train_resul), (test_input, test_resul)
    pass


def train_data1():
    batch_size = 1
    num_classes = 2
    epochs = 17
    (x_train, y_train), (x_test, y_test) = read_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    x_train *= 2
    x_test *= 2
    model = Sequential()
    # 确定输入层节点数及输入格式
    model.add(Dense(2, activation='relu', input_shape=(2,)))
    # 确定隐藏层节点数
    model.add(Dense(5, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])
    train_history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                              validation_data=(x_test, y_test))
    # model.save('D:/Pycharm/PythonProject/attempt/eye_model.h5')
    return train_history, model
    pass


def train_data2():
    batch_size = 1
    num_classes = 3
    epochs = 25
    (x_train, y_train), (x_test, y_test) = read_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    model = Sequential()
    # 确定输入层节点数及输入格式
    model.add(Dense(1, activation='relu', input_shape=(1,)))
    # 确定隐藏层节点数
    model.add(Dense(2, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])
    train_history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                              validation_data=(x_test, y_test))
    # model.save('D:/Pycharm/PythonProject/attempt/mouth_model.h5')
    return train_history, model
    pass


def print_history(train_history):
    # 绘制训练 & 验证的准确率值
    plt.plot(train_history.history['accuracy'])
    plt.plot(train_history.history['val_accuracy'])
    plt.plot(train_history.history['loss'])
    plt.plot(train_history.history['val_loss'])
    plt.title('Model accuracy&loss')
    plt.xlabel('Epoch')
    plt.legend(['Train_acc', 'Val_acc', 'Train_loss', 'Val_loss'])
    plt.show()


history, model = train_data2()
print(history.params)
print_history(history)  # 调用绘图函数
