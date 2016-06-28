# coding:utf-8
from file_io import *
from rnn import *
from datetime import datetime

# 使用梯度下降训练循环神经网络的参数
"""
model:模型
X_train,Y_train训练数据
learning_rate:梯度下降的步长
loss_threshold:损失函数的阈值 小于loss_threshold时结束
"""
def train_rnn_with_sgd(model, X_train, Y_train, learning_rate = 0.005, loss_threshold = 10.0):
    # 使用数组存储所有的损失值
    losses = []
    current_loss = 100000.0
    train_num = 0

    while current_loss > loss_threshold:
        train_num += 1

        print 'training......'
        # 使用梯度下降训练模型
        for i in range(len(X_train)):
            model.sgd_step(X_train[i], Y_train[i], learning_rate)

        # 计算损失值
        current_loss = model.caculate_loss(X_train, Y_train)
        losses.append(current_loss)
        time = datetime.now().strftime(u'%Y-%m-%d-%H-%M-%S')

        print u'时间：%s，次数：%d，损失值：%f' % (time, train_num, current_loss)

        # 如果当前损失值变大说明梯度步长太长
        if len(losses) > 1 and losses[-1] > losses[-2]:
            learning_rate = learning_rate * 0.8
            print u'设置学习步长为：%f' % learning_rate


######################################################################################

# 数据文件
filename = 'data\KAIST-1.trace'

# 读取训练数据和测试数据
X_train, Y_train, X_test, Y_test = read_file(filename, sample_len = 10, step = 1)

# 初始化一个rnn实例用于训练模型
rnn = TheanoRNN(hidden_him = 1000)

# 进行模型的训练
train_rnn_with_sgd(rnn, X_train, Y_train, learning_rate = 0.005, loss_threshold = 1.0)

print '矩阵U'
print rnn.U
print '--------------------------------------------------'
print '矩阵V'
print rnn.V
print '--------------------------------------------------'
print '矩阵W'
print rnn.W































