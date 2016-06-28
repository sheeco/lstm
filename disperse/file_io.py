# coding:utf-8
import numpy
import math


"""
读取文件代码，用于读取文件中的数据
一部分数据作为训练样本，一部分作为测试样本
一个文件对应一个节点的所有坐标
"""

"""
filename:读取文件名
sample_per:读取数据的样本所占的比例
sample_len：每个样本的长度（决定了循环神经网络的长度）
step:表示每隔多少个取一个坐标默认为1
"""
def read_file(filename, sample_per = 0.9, sample_len = 10, step = 1):
    # 读取文件信息并且得到坐标的个数
    lines = open(filename, 'rb').readlines()
    locations = []

    # 将文本文件变成数值存入到数组中
    for i in range(0, len(lines)):
        if 0 == i % step:
            locations.append([numpy.float32(x) for x in lines[i].split()][1:3])

    # 坐标的个数
    locations_size = len(locations)
    # 样本的个数
    sample_size = int(sample_per * locations_size)

    # 存储样本的输入和输出
    # sample_len个输入 一个输出
    sample_x = []
    sample_y = []

    # 读取样本数据
    for i in range(0, sample_size - sample_len + 1):
        sample_x.append(locations[i : i + sample_len])
        sample_y.append(locations[i + sample_len])

    # 存储测试的输入和输出
    test_x = []
    test_y = []

    # 读取测试数据 最后一个不包括在内
    # test_y为输出 每个测试用例只对应一个输出 即为预测的坐标
    for i in range(sample_size, locations_size - sample_len):
        test_x.append(locations[i : i + sample_len])
        test_y.append(locations[i + sample_len])

    return sample_x, sample_y, test_x, test_y

"""
将连续的数据进行离散化
disperse_radius网格的大小
"""
def read_disperse_postion(filename, sample_per = 0.9, sample_len = 10, step = 1, disperse_radius = 2.0):
    # 读取文件信息并且得到坐标的个数
    lines = open(filename, 'rb').readlines()
    lines_len = len(lines)

    # 申请一个数组 保存所有的坐标 int类型
    continue_pos = []

    # 将文本文件变成数值存入到数组中
    for i in range(0, lines_len):
        if 0 == i % step:
            continue_pos.append([float(x) for x in lines[i].split()][1:3])

    # 得到最大最小的x 最大最小的y
    min_x, min_y = numpy.min(continue_pos, axis=0)
    max_x, max_y = numpy.max(continue_pos, axis=0)

    disperse_matrix_width  = int(math.ceil(1.0 * (max_x - min_x) / disperse_radius) + 1.0)
    disperse_matrix_height = int(math.ceil(1.0 * (max_y - min_y) / disperse_radius) + 1.0)

    # 数组大小
    vector_dim = disperse_matrix_height * disperse_matrix_width

    # 定义01数组 离散化坐标
    # disperse_pos = numpy.zeros((len(continue_pos), vector_dim))
    disperse_pos = [int(0)] * len(continue_pos)

    for i in range(len(continue_pos)):
        disperse_pos[i] = int(round((continue_pos[i][1] - min_y) / disperse_radius)) * disperse_matrix_width + int(round((continue_pos[i][0] - min_x) / disperse_radius))

    # 坐标个数和样本个数
    disperse_size = len(disperse_pos)
    sample_size   = int(sample_per * disperse_size)

    # 样本点 数组
    X_train = []
    Y_train = []

    for i in range(0, sample_size - sample_len + 1):
        X_train.append(disperse_pos[i : i + sample_len])
        Y_train.append(disperse_pos[i + sample_len])

    # 存储测试的输入和输出 Y_test存储真实的坐标数据
    X_test = []
    Y_test = []
    for i in range(sample_size, disperse_size - sample_len):
        X_test.append(disperse_pos[i: i + sample_len])
        Y_test.append(continue_pos[i + sample_len])



    # 返回数组大小 和 样本点 测试点
    return vector_dim, X_train, Y_train, X_test, Y_test, min_x, min_y, disperse_matrix_width, disperse_matrix_height
























