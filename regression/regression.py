# coding:utf-8

from numpy import *
from data_io import *
import numpy as np

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

# 使用局部线性回归进行预测
# 使用时间作为输入 坐标作为输出

# 局部线性回归的系数计算如下：
# ww = (Xt W X)-1 Xt W Y
# x为输入 W 为权重 Y为输出
# 将前10个坐标作为原始数据 对应的[[0.0, 1.0], [1.0, 1.0]--[9.0, 1.0]]作为自变量
# 1.0为偏执
# 对应的x y坐标作为因变量 因此 需要预测的是[10.0, 1.0]对应的值

# X为自变量 predict_x为需要预测的自变量
def get_weights(xArr, predict_x, k = 2.0):
    # 转换成矩阵
    xMat = mat(xArr)
    # 样本点的个数
    m = shape(xMat)[0]
    # 权重 是一个m*m大小的矩阵
    weight = mat(eye((m)))

    for i in range(m):
        diffMat = predict_x - xMat[i]
        weight[i, i] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))

    xTx = xMat.T * weight * xMat

    if linalg.det(xTx) == 0.0:
        print '没有逆矩阵'
        return

    # 用于计算权重矩阵
    ws = xTx.I * xMat.T * weight

    return ws

# 计算的系数矩阵得到预测值
def get_predic_y(predict_x, ws, yArr):
    return asarray(mat(predict_x) * ws * mat(yArr))

def predict_position(filename, predict_x, weights, sample_len, out_filename, threshold = 100):
    positions = read_file(filename)
    predict_data_filename = 'predict\\' + filename + '.predict'

    # 原始数据和预测数据同时写入
    predict_data_file = open(predict_data_filename, 'a+')

    time = 300.0

    dises = []

    for i in range(len(positions) - sample_len):
        # 样本
        yArr = positions[i : i + sample_len]
        # 预测点
        predict_y = get_predic_y(predict_x, weights, yArr)
        # 欧式距离
        dis = math.sqrt(((positions[i + sample_len] - predict_y) ** 2).sum())

        dises.append(dis)

        # 将原始数据和预测数据写于文件
        predict_data_file.write(u'%f    %f    %f    %f\n' % (time, predict_y[0][0], predict_y[0][1], dis))

        time += 30.0

    predict_data_file.close()

    out_threshold = len([x for x in dises if x > threshold])

    print '测试文件：', filename
    print '平均距离：', np.mean(dises)
    print '最大距离：', np.max(dises)
    print '距离阈值：', threshold
    print '预测坐标个数：', len(dises)
    print '超过阈值个数：', out_threshold
    print '超过阈值比例：', 1.0 * out_threshold / len(dises)
    print '--------------------------------------------------------------------------------'

    # 将数据写入文件
    out_file = open(out_filename, 'a+')
    out_file.write(u'测试文件：%s\n' % (filename) )
    out_file.write(u'平均距离：%f\n' % (np.mean(dises)))
    out_file.write(u'最大距离：%f\n' % (np.max(dises)))
    out_file.write(u'距离阈值：%f\n' % (threshold))
    out_file.write(u'预测坐标个数：%d\n' % (len(dises)))
    out_file.write(u'超过阈值个数：%d\n' % (out_threshold))
    out_file.write(u'超过阈值比例：%f\n' % (1.0 * out_threshold / len(dises)))
    out_file.write(u'-------------------------------------------------------------------------\n')
    out_file.close()






































