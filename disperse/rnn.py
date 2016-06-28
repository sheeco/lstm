# coding:utf-8
import numpy as np
from theano import *
import theano.tensor as T
import operator

# 使用theano编写rnn 参考：https://github.com/dennybritz/rnn-tutorial-rnnlm/blob/master/rnn_theano.py
# s(t) = tanh(U*x(t) + W * s(t-1))
# o(t) = V * s(t)
# x(t)输入
# s(t)隐藏层
# o(t)输出
class TheanoRNN:
    # location_dim坐标维度
    # hidden_dim隐藏层的维度
    # bptt_truncate使用bptt算法训练参数时 梯度向前传播的最远距离
    def __init__(self, pos_dim, hidden_him = 100, bptt_truncate = 4):

        # 存储参数
        self.pos_dim       = pos_dim
        self.hidden_dim    = hidden_him
        self.bptt_truncate = bptt_truncate

        # 随机初始化需要的矩阵 维度根据输入 隐藏层 和 输出进行判定
        U = np.random.uniform(-np.sqrt(1.0 / pos_dim),    np.sqrt(1.0 / pos_dim),    (hidden_him, pos_dim))
        V = np.random.uniform(-np.sqrt(1.0 / hidden_him), np.sqrt(1.0 / hidden_him), (pos_dim, hidden_him))
        W = np.random.uniform(-np.sqrt(1.0 / hidden_him), np.sqrt(1.0 / hidden_him), (hidden_him, hidden_him))

        # 定义成shared类型 需要在迭代中多次使用
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))

        # 初始化需要的各种函数表达式
        self.__theano_build__()

    # 初始化变量表达式 和 公式表达式
    def __theano_build__(self):
        U, V, W = self.U, self.V, self.W

        # x为一个整数数组 比如[1, 46, 67]表示对应的坐标点的索引
        # y为预测值 一个整数
        x = T.ivector('x')
        y = T.iscalar('y')

        # 定义一个函数表示每一步向前计算的过程
        # x_t序列参数 表示当前循环神经网络的输入和 theano.scan迭代的次数
        # s_t_prev存储上一层的隐藏层 也表示theano.scan的上一次迭代的结果
        # UVW不变量 每次迭代均不变
        # threan.scan用于输出一系列的“表达式！！！！”
        def forward_prop_step(x_t, s_t_prev, U, W):
            s_t = T.tanh(U[:, x_t] + W.dot(s_t_prev))

            return s_t

        # scan的函数(fn)的参数序列为：sequences (if any), prior result(s) (if needed), non-sequences (if any)
        # 数列为x
        # 参数接受上次输出为s_t_prev(函数有两个输出第一个设为None)
        # 不变序列UVW
        # truncate_gradient表示梯度节点向前传播的次数
        s, updates = theano.scan(
            forward_prop_step,
            sequences = x,
            outputs_info = dict(initial = T.zeros(self.hidden_dim, dtype = theano.config.floatX)),
            non_sequences = [U, W],
            truncate_gradient=self.bptt_truncate,
            strict=True)

        # 只需要最后一个隐藏层的值
        # o为一个pos_dim长度的向量
        o = T.nnet.softmax(V.dot(s[-1]))

        # 预测值 返回向量中最大值的下表 即为预测值
        prediction = T.argmax(o)

        # 使用熵作为损失函数
        # o是一个矩阵 首先选取第一维度的第一个数据
        error = -T.sum(T.log(o[0,y]))

        # 计算梯度表达式
        dU = T.grad(error, U)
        dV = T.grad(error, V)
        dW = T.grad(error, W)

        # 计算预测值
        self.forward_propagation = function([x], prediction)

        # 计算损失
        self.loss = function([x, y], error)

        # 使用bptt算法计算梯度
        self.bptt = function([x, y], [dU, dV, dW])

        # 梯度下降更新参数
        # 学习进度
        learning_rate = T.scalar('learning_rate')
        # 梯度下降更新新的参数值
        self.sgd_step = function([x, y, learning_rate], [],
                                 updates = [(self.U, self.U - learning_rate * dU),
                                            (self.V, self.V - learning_rate * dV),
                                            (self.W, self.W - learning_rate * dW)])

    # 计算损失函数
    # 使用熵作为损失函数
    def caculate_loss(self, X, Y):
        max_loss = 0
        min_loss = 9999999
        avg_loss = 0

        for i in range(len(X)):
            loss = self.loss(X[i], Y[i])
            avg_loss += loss
            max_loss = max(max_loss, loss)
            min_loss = min(min_loss, loss)

        return max_loss, min_loss, 1.0 * avg_loss / len(X)





# 定义一个函数用于检测上面的bptt算法计算的梯度是否正确
# (f(x+h) - f(x-h))/(2*h) h步长
# model为一个训练模型
# x y样本
# error_threshold 错误阈值
def gradient_check_theano(model, x, y, h = 0.001, error_threshold = 0.01):
    # bptt算法的梯度传播调到最大 用于计算准确的梯度
    model.bptt_truncate = 1000
    # 得到模型计算的梯度
    bptt_gradients = model.bptt(x, y)

    # 便利model所有的参数 然后修改其对应的值 来计算梯度
    # List of all parameters we want to chec.
    model_parameters = ['U', 'V', 'W']
    i = 0
    total = 0
    # Gradient check for each parameter
    for pidx, pname in enumerate(model_parameters):
        parameter_T = operator.attrgetter(pname)(model)
        parameter = parameter_T.get_value()
        print "计算参数： %s 的梯度，个数为 %d." % (pname, np.prod(parameter.shape))
        total += np.prod(parameter.shape)
        print parameter.shape

        # 遍历
        it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            # 存储原始梯度
            original_value = parameter[ix]
            # 修改变量值计算真实梯度
            parameter[ix] = original_value + h
            parameter_T.set_value(parameter)
            gradplus = model.avg_loss(x, y)

            parameter[ix] = original_value - h
            parameter_T.set_value(parameter)
            gradminus = model.avg_loss(x, y)

            estimated_gradient = (gradplus - gradminus) / (2 * h)

            parameter[ix] = original_value
            parameter_T.set_value(parameter)

            # bptt计算的梯度
            backprop_gradient = bptt_gradients[pidx][ix]

            # calculate The relative error: (|x - y|/(|x| + |y|))
            relative_error = np.abs(backprop_gradient - estimated_gradient) / (np.abs(backprop_gradient) + np.abs(estimated_gradient))

            if relative_error > error_threshold:
                i += 1
                # print "梯度错误: 参数=%s 索引=%s" % (pname, ix)
                # print "+h Loss: %f" % gradplus
                # print "-h Loss: %f" % gradminus
                # print "估算的梯度: %f" % estimated_gradient
                # print "bptt对应的梯度: %f" % backprop_gradient
                # print "相关错误: %f" % relative_error
                # return

            it.iternext()

        print "参数 %s 梯度检测完毕" % (pname)

    print 'i:%d' % i
    print 'total:%d' % total























































