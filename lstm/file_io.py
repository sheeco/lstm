# coding:utf-8
import numpy
# import math
import os  # 文件夹操作
import config


def read_triples_from_file(filename):
    """
    从单个文件读入 (time, x, y) 的三元组序列，放入返回 list。
    注意：暂无格式检查，文件中每一行必须对应一次定位采样，格式为 'time x y'
    :param filename: string 文件名
    :return: list[:行数, :3]
    """

    lines = open(filename, 'rb').readlines()
    triples = []  # (time, x, y) 的三元组列表

    # 从每一行读入三个数值并存入列表中的一行
    for i in range(0, len(lines)):
        triples.append([numpy.float32(x) for x in lines[i].split()][0:3])

    # n_triples = len(triples)

    return triples


def read_traces_from_path(path):
    # todo exception handler for invalid path
    # todo add filename filter
    """
    对给定目录下的所有文件，读取轨迹序列，放入返回 dict。
    注意：暂无扩展名检查
    :param path: string 指定路径
    :return: dict{轨迹文件名 string: 采样三元组 list[:行数, :3]}
    """

    dict_traces = {}

    if not os.path.exists(path):
        print 'Invalid Path: ' + path
        return
    else:
        list_subdir = os.listdir(path)
        list_files = [subdir for subdir in list_subdir if os.path.isfile(path + subdir)]

        for filename in list_files:
            node_identifier, _ = os.path.splitext(filename)
            temp_trace = read_triples_from_file(path + filename)
            # if node_name.isdigit():
            #     traces[int(node_name)] = temp_trace
            # else:
            #     traces[node_name] = temp_trace
            dict_traces[node_identifier] = temp_trace

    return dict_traces


def generate_input_sequences(dict_traces, node_identifiers, i_line_entry, with_target=True):
    """
    从指定节点的轨迹中，以指定位置作为起始 instant，为网络的训练/测试生成输入序列（及目标序列），放入返回的 array。
    注意：如果超出最大采样数，将返回 2 个空值，因此需要对返回值进行验证之后再使用
    :param dict_traces: 由 @read_traces_from_path 返回的 dict{轨迹文件名 string: 采样三元组 list}
    :param node_identifiers: 用于指定所选节点集合的 array/list[string 节点标识符]
    :param i_line_entry: int 指定序列起始 instant 在轨迹文件中对应的行号（由 0 开始）
    :param with_target: bool 是否返回学习目标，默认 True；如果为 False，第二个返回值即为空
    :return: array[:所选节点数, :LENGTH_SEQUENCE_INPUT, :2], array[:所选节点数, :LENGTH_SEQUENCE_OUTPUT， :2]
    """

    # 仅选中 node_identifiers 中指定的节点的轨迹
    if len(node_identifiers) > 0:
        nodes_requested = node_identifiers
    # 选中所有节点的轨迹
    else:
        nodes_requested = dict_traces.keys()
    # todo exception handler if no value found by given key
    dict_traces_requested = {node_id: dict_traces[node_id] for node_id in nodes_requested}

    # 计算不同节点轨迹长度的最小值
    min_len_trace = min([len(trace) for trace in dict_traces_requested.values()])
    # 以 min_len_trace 为最后一行，对齐截断
    # 并将选中 dict 中的 value 转换成 array，顺序由给定的 node_identifiers 决定
    traces_requested = numpy.array([numpy.array(trace)[:min_len_trace, :] for trace in dict_traces_requested.values()],
                                   dtype=numpy.float64)

    # 不使用 time 作为输入元组的一部分，删除第 0 列
    traces_requested = traces_requested[:, :, 1:]

    begin_line_input = i_line_entry
    end_line_input = begin_line_input + config.LENGTH_SEQUENCE_INPUT
    begin_line_target = end_line_input
    end_line_target = begin_line_target + config.LENGTH_SEQUENCE_OUTPUT

    # 如果超出采样数，返回 [], []
    sequences_input = numpy.array([])
    sequences_target = numpy.array([])
    end_line_traces = traces_requested.shape[1]
    if end_line_input >= end_line_traces or end_line_target > end_line_traces:
        return sequences_input, sequences_target

    sequences_input = traces_requested[:, begin_line_input: end_line_input, :]
    if with_target:
        sequences_target = traces_requested[:, begin_line_target: end_line_target, :]

    return sequences_input, sequences_target


# """
# filename:读取文件名
# sample_per:读取数据的样本所占的比例
# sample_len：每个样本的长度（决定了循环神经网络的长度）
# step:表示每隔多少个取一个坐标默认为1
# """
# def read_file(filename, sample_per = 0.9, sample_len = 10, step = 1):
#     # 读取文件信息并且得到坐标的个数
#     lines = open(filename, 'rb').readlines()
#     locations = []
#
#     # 将文本文件变成数值存入到数组中
#     for i in range(0, len(lines)):
#         if 0 == i % step:
#             locations.append([numpy.float32(x) for x in lines[i].split()][1:3])
#
#     # 坐标的个数
#     locations_size = len(locations)
#     # 样本的个数
#     sample_size = int(sample_per * locations_size)
#
#     # 存储样本的输入和输出
#     # sample_len个输入 一个输出
#     sample_x = []
#     sample_y = []
#
#     # 读取样本数据
#     for i in range(0, sample_size - sample_len + 1):
#         sample_x.append(locations[i : i + sample_len])
#         sample_y.append(locations[i + sample_len])
#
#     # 存储测试的输入和输出
#     test_x = []
#     test_y = []
#
#     # 读取测试数据 最后一个不包括在内
#     # test_y为输出 每个测试用例只对应一个输出 即为预测的坐标
#     for i in range(sample_size, locations_size - sample_len):
#         test_x.append(locations[i : i + sample_len])
#         test_y.append(locations[i + sample_len])
#
#     return sample_x, sample_y, test_x, test_y


# """
# 将连续的数据进行离散化
# disperse_radius网格的大小
# """
#
#
# def read_disperse_postion(filename, sample_per=0.9, sample_len=10, step=1, disperse_radius=2.0):
#     # 读取文件信息并且得到坐标的个数
#     lines = open(filename, 'rb').readlines()
#     lines_len = len(lines)
#
#     # 申请一个数组 保存所有的坐标 int类型
#     continue_pos = []
#
#     # 将文本文件变成数值存入到数组中
#     for i in range(0, lines_len):
#         if 0 == i % step:
#             continue_pos.append([float(x) for x in lines[i].split()][1:3])
#
#     # 得到最大最小的x 最大最小的y
#     min_x, min_y = numpy.min(continue_pos, axis=0)
#     max_x, max_y = numpy.max(continue_pos, axis=0)
#
#     disperse_matrix_width = int(math.ceil(1.0 * (max_x - min_x) / disperse_radius) + 1.0)
#     disperse_matrix_height = int(math.ceil(1.0 * (max_y - min_y) / disperse_radius) + 1.0)
#
#     # 数组大小
#     vector_dim = disperse_matrix_height * disperse_matrix_width
#
#     # 定义01数组 离散化坐标
#     # disperse_pos = numpy.zeros((len(continue_pos), vector_dim))
#     disperse_pos = [int(0)] * len(continue_pos)
#
#     for i in range(len(continue_pos)):
#         disperse_pos[i] = int(round((continue_pos[i][1] - min_y) / disperse_radius)) * disperse_matrix_width + int(
#             round((continue_pos[i][0] - min_x) / disperse_radius))
#
#     # 坐标个数和样本个数
#     disperse_size = len(disperse_pos)
#     sample_size = int(sample_per * disperse_size)
#
#     # 样本点 数组
#     X_train = []
#     Y_train = []
#
#     for i in range(0, sample_size - sample_len + 1):
#         X_train.append(disperse_pos[i: i + sample_len])
#         Y_train.append(disperse_pos[i + sample_len])
#
#     # 存储测试的输入和输出 Y_test存储真实的坐标数据
#     X_test = []
#     Y_test = []
#     for i in range(sample_size, disperse_size - sample_len):
#         X_test.append(disperse_pos[i: i + sample_len])
#         Y_test.append(continue_pos[i + sample_len])
#
#     # 返回数组大小 和 样本点 测试点
#     return vector_dim, X_train, Y_train, X_test, Y_test, min_x, min_y, disperse_matrix_width, disperse_matrix_height


def test_file_io():
    demo_list_triples = read_triples_from_file('data/2.trace')
    dict_all_traces = read_traces_from_path('data/')
    in_sequences, out_sequences = generate_input_sequences(dict_all_traces, [], 0, True)
    in_sequences, out_sequences = generate_input_sequences(dict_all_traces, ['1', '2'], 0, True)
    in_sequences, out_sequences = generate_input_sequences(dict_all_traces, numpy.array(['1', '2']), 0, True)
    in_sequences, out_sequences = generate_input_sequences(dict_all_traces, [], 1, True)
    in_sequences, out_sequences = generate_input_sequences(dict_all_traces, [], 0, False)

    return


if __name__ == '__main__':
    test_file_io()
