# coding:utf-8
from file_io import *
from rnn import *
from datetime import datetime

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

# 使用梯度下降训练循环神经网络的参数
"""
model:模型
X_train,Y_train训练数据
learning_rate:梯度下降的步长
loss_threshold:损失函数的阈值 小于loss_threshold时结束
"""
def train_rnn_with_sgd(input_file_name, learning_rate = 0.005, threshold = 0.0001):
    # 离散半径
    disperse_radius = 10.0

    # 定义两个文件 用于存储中间数据
    train_output_file_name = input_file_name + str(disperse_radius) + '.train'
    test_ouput_file_name   = input_file_name + str(disperse_radius) + '.test'

    # 保存权值的文件
    UVW_file_name = input_file_name + str(disperse_radius) + '.UVW'

    # 读取训练数据和测试数据
    vector_dim, X_train, Y_train, X_test, Y_test, min_x, min_y, disperse_matrix_width, disperse_matrix_height = read_disperse_postion(input_file_name, sample_per = 0.8, sample_len = 10, step = 1, disperse_radius = disperse_radius)

    # 实例化一个循环神经网络类
    rnn = TheanoRNN(pos_dim = vector_dim, hidden_him = 100, bptt_truncate = 4)

    losses = []
    train_num = 0
    loss_threshold = 10000.0

    while loss_threshold > threshold:
        # 训练次数
        train_num += 1

        # 标记
        print '%s is training....' % input_file_name

        # 使用梯度下降训练模型
        for i in range(len(X_train)):
            # print X_train[i], Y_train[i]
            rnn.sgd_step(X_train[i], Y_train[i], learning_rate)

        # 计算损失值
        max_loss, min_loss, current_loss = rnn.caculate_loss(X_train, Y_train)
        losses.append(current_loss)
        time = datetime.now().strftime(u'%Y-%m-%d-%H-%M-%S')

        # 显示各种数据
        print u'训练数据：%s，时间：%s，次数：%d，平均损失熵：%f，最大损失熵：%f' % (input_file_name, time, train_num, current_loss, max_loss)

        # 将数据写入文件
        train_output_file = open(train_output_file_name, 'a+')
        train_output_file.write(u'时间：%s，次数：%d，平均损失熵：%f，最大损失熵：%f\n' % (time, train_num, current_loss, max_loss))
        train_output_file.close()

        if len(losses) > 1:
            # 损失值得变化
            loss_threshold = abs(losses[-1] - losses[-2])

            # 如果当前损失值变大说明梯度步长太长
            if losses[-1] > losses[-2]:
                learning_rate = learning_rate * 0.6
                print u'设置学习步长为：%f' % learning_rate

    # 存储最远的平均距离
    avg_dis = 0.0
    max_dis = 0.0

    for i in range(len(X_test)):
        prediction = rnn.forward_propagation(X_test[i])
        # 预测坐标
        predict_y = [1.0 * (prediction % disperse_matrix_width) * disperse_radius, 1.0 * (prediction / disperse_matrix_width) * disperse_radius]
        # 距离
        dis = (((np.array(predict_y) - np.array(Y_test[i])) ** 2).sum()) ** 0.5

        avg_dis += dis
        max_dis = max(max_dis, dis)

    avg_dis = 1.0 * avg_dis / len(X_test)

    test_ouput_file = open(test_ouput_file_name, 'a+')
    test_ouput_file.write(u'平均欧式距离：%f\n' % (avg_dis))
    test_ouput_file.write(u'最大欧氏距离：%f' % (max_dis))
    test_ouput_file.close()
    print u"保存测试数据到文件：%s" % (test_ouput_file_name)
    print u"平均距离：%f，最大距离：%f" % (avg_dis, max_dis)

    # 将权值数据保存到文件
    U, V, W = rnn.U.get_value(), rnn.V.get_value(), rnn.W.get_value()
    np.savez(UVW_file_name, U=U, V=V, W=W)
    print u"保存参数到文件：%s." % UVW_file_name



######################################################################################

# 数据文件
filename = "Statefair\\1.trace"

# 训练测试
train_rnn_with_sgd(filename, learning_rate = 0.005, threshold = 0.00001)




























