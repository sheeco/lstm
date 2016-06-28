# coding:utf-8
# 读取文件坐标
def read_file(filename, step = 1):
    # 读取文件信息并且得到坐标的个数
    lines = open(filename, 'rb').readlines()
    postions = []

    # 将文本文件变成数值存入到数组中
    for i in range(len(lines)):
        if 0 == i % step:
            postions.append([float(x) for x in lines[i].split()][1:3])

    return postions
