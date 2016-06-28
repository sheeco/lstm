# coding:utf-8
from numpy import *
from regression import *
from data_io import *

# 每个坐标和前sample_len有关系
sample_len = 10

# 自定义输入为自变量 以时间轴作为自变量
xArr = []
for i in range(sample_len):
    xArr.append([i, 1.0])

# 需要预测的自变量
predict_x = [10, 1.0]

# 用于计算权值
weights = get_weights(xArr, predict_x, k = 1.0)

# 距离的阈值
dis_threshold = 100

# 预测数据输出文件
out_filename = 'out.data'

filenames = []
for i in range(1, 10):
    filenames.append('Statefair\\%d.trace'% i)

for i in range(1, 85):
    filenames.append('KAIST\\%d.trace' % i)

for filename in filenames:
    predict_position(filename, predict_x, weights, sample_len, out_filename, threshold = dis_threshold)
















