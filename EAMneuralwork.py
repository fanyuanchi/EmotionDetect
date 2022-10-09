import keras
from keras.models import load_model
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adadelta, adam
import matplotlib.pyplot as plt


# 装载训练集、测试集数据和标签
# train_data_path是训练集数据文件路径，train_label_path是训练集标签文件路径
# test_data_path是测试集数据文件路径，test_label_path是测试集标签文件路径
def load_data(train_data_path, train_label_path, test_data_path, test_label_path):
    train_data_file = open(train_data_path, "r")
    train_result_file = open(train_label_path, "r")
    test_data_file = open(test_data_path, "r")
    test_result_file = open(test_label_path, "r")
    # 依次打开训练集、测试集数据和标签文件
    train_data_list = train_data_file.readlines()
    train_result_list = train_result_file.readlines()
    test_data_list = test_data_file.readlines()
    test_result_list = test_result_file.readlines()
    # 按行读入文件内容
    train_data_file.close()
    train_result_file.close()
    test_data_file.close()
    test_result_file.close()
    # 关闭文件
    train_inputs = []
    train_result = []
    test_inputs = []
    test_result = []
    # 初始化空列表
    for record in train_data_list:
        all_values = record.split(',')
        train_inputs.append(all_values[:])
    pass
    for record in train_result_list:
        all_values = record.split(',')
        train_result.append(all_values[:])
    pass
    for record in test_data_list:
        all_values = record.split(',')
        test_inputs.append(all_values[:])
    pass
    for record in test_result_list:
        all_values = record.split(',')
        test_result.append(all_values[:])
    pass
    # 将数据以“，”分隔后插入列表
    train_input = np.array(train_inputs)
    train_result = np.array(train_result)
    test_input = np.array(test_inputs)
    test_result = np.array(test_result)
    # 将列表转化为数组矩阵
    return (train_input, train_result), (test_input, test_result)
    pass


# 构建并训练EAR值分类深度学习神经网络
def train_EAR(train_data_path, train_label_path, test_data_path, test_label_path, model_path):
    # 分批训练集每一批训练个数
    batch_size = 1
    # 分类个数，此为二分类
    num_classes = 2
    # 训练轮次
    epochs = 17
    # 装载训练集、测试集数据和标签
    (x_train, y_train), (x_test, y_test) = load_data(train_data_path, train_label_path, test_data_path, test_label_path)
    # 将数据集转化为32未浮点数
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # 将标签集转化为二分类支持格式
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    # 对数据集进行适当放缩提高训练精确度
    x_train *= 2
    x_test *= 2

    model = Sequential()
    # 确定输入层节点数、激活函数及输入格式
    model.add(Dense(2, activation='relu', input_shape=(2,)))
    # 确定隐藏层节点数及激活函数
    model.add(Dense(5, activation='relu'))
    # 确定输出层节点数及输出挤压函数
    model.add(Dense(num_classes, activation='softmax'))
    # model.summary()
    # 确定优化器，包括损失函数（交叉熵损失函数）、优化函数及训练表达函数
    model.compile(loss='categorical_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])
    train_history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                              validation_data=(x_test, y_test))
    # model.save(model_path)
    return train_history, model
    pass


# 构建并训练MAR值分类深度学习神经网络
def train_MAR(train_data_path, train_label_path, test_data_path, test_label_path, model_path):
    batch_size = 1
    num_classes = 3
    epochs = 25
    (x_train, y_train), (x_test, y_test) = load_data(train_data_path, train_label_path, test_data_path, test_label_path)
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
    # model.save(model_path)
    return train_history, model
    pass


# 构建并训练EBAR值分类深度学习神经网络
def train_EBAR(train_data_path, train_label_path, test_data_path, test_label_path, model_path):
    batch_size = 16
    num_classes = 1
    epochs = 500
    (x_train, y_train), (x_test, y_test) = load_data(train_data_path, train_label_path, test_data_path, test_label_path)
    x_train = x_train.astype('float32')
    y_train = y_train.astype('float32')
    x_test = x_test.astype('float32')
    y_test = y_test.astype('float32')
    # y_train = keras.utils.to_categorical(y_train, num_classes)
    # y_test = keras.utils.to_categorical(y_test, num_classes)

    # x_train *= 4
    # x_test *= 4

    model = Sequential()
    # 确定输入层节点数及输入格式
    model.add(Dense(16, activation='sigmoid', input_shape=(1,)))
    # 确定隐藏层节点数
    model.add(Dense(5, activation='relu'))
    model.add(Dense(num_classes))
    # model.summary()
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])
    train_history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                              validation_data=(x_test, y_test))
    # model.save(model_path)
    return train_history, model
    pass


# 绘制深度学习神经网络训练结果图像
def print_history1(train_history):
    # 绘制训练 & 验证(回归)的准确率值
    plt.plot(train_history.history['accuracy'])
    plt.plot(train_history.history['val_accuracy'])
    # 绘制训练 & 验证(回归)的损失率值
    plt.plot(train_history.history['loss'])
    plt.plot(train_history.history['val_loss'])

    plt.title('Model accuracy&loss')
    plt.xlabel('Epoch')
    plt.legend(['Train_acc', 'Val_acc', 'Train_loss', 'Val_loss'])
    plt.show()
    pass


def print_history2(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'g.', label='traning loss')
    plt.plot(epochs, val_loss, 'b.', label='validation loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    pass


def print_history3(x_train, y_train, model_normal, model_angry):
    predictions_normal = []
    predictions_angry = []
    x_train = x_train.astype('float32')
    y_train = y_train.astype('float32')
    for i in range(0, len(x_train)):
        prediction_normal = model_normal.predict(x_train[i])
        prediction_angry = model_angry.predict(x_train[i])
        predictions_normal.append(prediction_normal)
        predictions_angry.append(prediction_angry)

    fig = plt.figure()  # 创建画布
    ax = fig.add_subplot(111)

    x_train = list(np.array(x_train).flatten())
    y_train = list(np.array(y_train).flatten())
    # p1 = ax.scatter(x_train, y_train, marker='.', color='black', s=8)
    p2 = ax.scatter(x_train, predictions_normal, marker='.', color='red', s=8)
    p3 = ax.scatter(x_train, predictions_angry, marker='.', color='blue', s=8)
    plt.show()  # 显示散点图
    pass


# 装载模型
def load_neural(neural_path):
    return load_model(neural_path)
    pass
