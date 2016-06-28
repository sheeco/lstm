# coding:utf-8
import numpy


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





















