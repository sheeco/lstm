# coding:utf-8
import numpy
import os  # 文件夹操作
import config
from utils import *


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

    return numpy.array(triples, dtype=numpy.float32)


def read_traces_from_path(path):
    # todo add filename filter
    """
    对给定目录下的所有文件，读取轨迹序列，放入返回 dict。
    注意：暂无扩展名检查
    :param path: string 指定路径
    :return: dict{轨迹文件名 string: 采样三元组 list[:行数, :3]}
    """

    dict_traces = {}

    if not os.path.exists(path):
        raise IOError('read_traces_from_path @ sample: \n\tInvalid Path: ' + str(path))
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


def get_range(dict_traces):
    range = numpy.zeros((2, 2))
    temp = tuple(dict_traces.itervalues())
    samples = numpy.concatenate(tuple(dict_traces.itervalues()), axis=0)
    range[0, 0] = numpy.min(samples[:, 1])
    range[0, 1] = numpy.max(samples[:, 1])
    range[1, 0] = numpy.min(samples[:, 2])
    range[1, 1] = numpy.max(samples[:, 2])

    return range


def pan_to_positive(dict_traces):
    """

    :param dict_traces:
    :param range:
    """

    range = get_range(dict_traces)
    stride = - range[:, 0]
    stride = numpy.ceil(stride)
    for trace in dict_traces.values():
        for itriple in xrange(len(trace)):
            trace[itriple, 1:3] = trace[itriple, 1:3] + stride
    return dict_traces


def load_sample_for_nodes(dict_traces, filter_nodes, idx_line_entry, with_target=True):
    """
    指定起始位置，为单次训练/测试生成所有节点的输入序列（及目标序列）。
    注意：如果超出最大采样数，将返回 None，因此需要对返回值进行验证之后再使用
    :param dict_traces: 由 @read_traces_from_path 返回的 dict{轨迹文件名 string: 采样三元组 list}
    :param filter_nodes: 用于指定所选节点集合的 array/list[string 节点标识符]
    :param idx_line_entry: int 指定序列起始 instant 在轨迹文件中对应的行号（由 0 开始）
    :param with_target: bool 是否返回学习目标，默认 True；如果为 False，第二个返回值即为 None
    :return: 时序，采样输入，学习目标
    :format: [:LENGTH_SEQUENCE_INPUT], [:所选节点数, :LENGTH_SEQUENCE_INPUT, :DIMENSION_SAMPLE], [:所选节点数, :LENGTH_SEQUENCE_OUTPUT， :DIMENSION_SAMPLE]
    """

    # 仅选中 node_identifiers 中指定的节点的轨迹
    if type(filter_nodes) == int and filter_nodes > 0:
        nodes_requested = dict_traces.keys()[:filter_nodes]
    elif type(filter_nodes) == list and len(filter_nodes) > 0:
        nodes_requested = filter_nodes
    # 选中所有节点的轨迹
    else:
        nodes_requested = dict_traces.keys()
    try:
        dict_traces_requested = {node_id: dict_traces[node_id] for node_id in nodes_requested}
    except KeyError:
        raise KeyError("generate_input_sequences @ sample: \n\tNo value found by given key.")

    # 计算不同节点轨迹长度的最小值
    min_len_trace = min([len(trace) for trace in dict_traces_requested.values()])
    # 以 min_len_trace 为最后一行，对齐截断
    # 并将选中 dict 中的 value 转换成 array，顺序由给定的 node_identifiers 决定
    traces_requested = numpy.array([trace[:min_len_trace, :] for trace in dict_traces_requested.values()],
                                   dtype=numpy.float32)
    # 提取时序
    instants = traces_requested[:, :, :1]
    instants = instants[0]
    instants = instants.sum(1)

    if config.DIMENSION_SAMPLE == 2:
        # 不使用 time 作为输入元组的一部分，删除第 0 列
        traces_requested = traces_requested[:, :, 1:]

    begin_line_input = idx_line_entry
    end_line_input = begin_line_input + config.LENGTH_SEQUENCE_INPUT
    begin_line_target = end_line_input
    end_line_target = begin_line_target + config.LENGTH_SEQUENCE_OUTPUT

    # 如果超出采样数，返回 [], [], []
    sequences_instants = numpy.zeros((0))
    sequences_input = numpy.zeros((0, 0, 0))
    sequences_target = numpy.zeros((0, 0, 0))
    end_line_traces = traces_requested.shape[1]
    if end_line_input >= end_line_traces or end_line_target > end_line_traces:
        return None, None, None

    sequences_instants = instants[begin_line_input: end_line_input]
    sequences_input = traces_requested[:, begin_line_input: end_line_input, :]

    sequences_target = traces_requested[:, begin_line_target: end_line_target, :] if with_target else None

    return sequences_instants, sequences_input, sequences_target


def load_batch_for_nodes(dict_traces, size_batch, filter_nodes, idx_line_entry, with_target=True):
    """
    指定起始位置及批大小，为单批次的训练/测试生成所有节点的输入序列（及目标序列）。
    注意：如果超出最大采样数，将返回 2 个 None，因此需要对返回值进行验证之后再使用
    :param dict_traces: 由 @read_traces_from_path 返回的 dict{轨迹文件名 string: 采样三元组 list}
    :param size_batch: int 批大小
    :param filter_nodes: 用于指定所选节点集合的 array/list[string 节点标识符]
    :param idx_line_entry: int 指定序列起始 instant 在轨迹文件中对应的行号（由 0 开始）
    :param with_target: bool 是否返回学习目标，默认 True；如果为 False，第二个返回值即为 None
    :return: 时序，采样输入，学习目标
    # :format: [size_batch, LENGTH_SEQUENCE_INPUT], [size_batch, N_NODES, LENGTH_SEQUENCE_INPUT, DIMENSION_SAMPLE], [size_batch, N_NODES, LENGTH_SEQUENCE_OUTPUT, DIMENSION_SAMPLE]
    :format: [size_batch, LENGTH_SEQUENCE_INPUT], [N_NODES, size_batch, LENGTH_SEQUENCE_INPUT, DIMENSION_SAMPLE], [N_NODES, size_batch, LENGTH_SEQUENCE_OUTPUT, DIMENSION_SAMPLE]
    """
    # todo add boolean arg redundant_batch

    batch_instants = numpy.zeros((0, 0))
    batch_input = numpy.zeros((0, 0, 0, 0))
    batch_target = numpy.zeros((0, 0, 0, 0))
    for i in range(size_batch):
        sample_instants, sample_input, sample_target = load_sample_for_nodes(dict_traces, filter_nodes, idx_line_entry + i,
                                                            with_target)
        if sample_input is not None:
            shape_in = sample_input.shape

            batch_instants = numpy.resize(batch_instants, (i + 1, shape_in[1]))
        #     batch_input = numpy.resize(batch_input, (i + 1, shape_in[0], shape_in[1], shape_in[2]))
        #     batch_input[i] = sample_input
        #     # batch_input = numpy.concatenate((batch_input, sample_input), 1) if len(batch_input) > 0 else sample_input
        #     if with_target:
        #         shape_target = sample_target.shape
        #         batch_target = numpy.resize(batch_target, (i + 1, shape_target[0], shape_target[1], shape_target[2]))
        #         batch_target[i] = sample_target

            batch_input = numpy.resize(batch_input, (shape_in[0], i + 1, shape_in[1], shape_in[2]))
            batch_instants[i] = sample_instants
            for inode in range(shape_in[0]):
                batch_input[inode, i] = sample_input[inode]
            if with_target:
                shape_target = sample_target.shape
                batch_target = numpy.resize(batch_target, (shape_target[0], i + 1, shape_target[1], shape_target[2]))
                for inode in range(shape_target[0]):
                    batch_target[inode, i] = sample_target[inode]

        elif len(batch_input) == 0:
            return None, None, None
        elif config.STRICT_BATCH_SIZE:
            warn("An insufficient batch of %s samples is discarded." % batch_input.shape[1])
            return None, None, None
        elif not config.STRICT_BATCH_SIZE:
            warn("Insufficient batch. Only  %s  samples are left." % batch_input.shape[1])
            break
        else:
            raise RuntimeError("load_batch_for_nodes @ sample: \n\tUnexpected access of this block.")
    if not with_target:
        batch_target = None
    return batch_instants, batch_input, batch_target


__all__ = ["read_traces_from_path",
           "pan_to_positive",
           "load_batch_for_nodes"]


def test_sample():
    demo_list_triples = read_triples_from_file('res/trace/2.trace')
    print numpy.shape(demo_list_triples)
    dict_all_traces = read_traces_from_path(config.PATH_TRACE_FILES)
    t_sample, in_sample, target_sample = load_sample_for_nodes(dict_all_traces, [], 0, True)
    print numpy.shape(t_sample), numpy.shape(in_sample), numpy.shape(target_sample)
    t_sample, in_sample, target_sample = load_sample_for_nodes(dict_all_traces, ['1', '2'], 0, True)
    print numpy.shape(t_sample), numpy.shape(in_sample), numpy.shape(target_sample)
    t_sample, in_sample, target_sample = load_sample_for_nodes(dict_all_traces, [], 1, True)
    print numpy.shape(t_sample), numpy.shape(in_sample), numpy.shape(target_sample)
    t_sample, in_sample, target_sample = load_sample_for_nodes(dict_all_traces, [], 0, False)
    print numpy.shape(t_sample), numpy.shape(in_sample), numpy.shape(target_sample)
    t_batch, in_batch, target_batch = load_batch_for_nodes(dict_all_traces, 31, [], 0, True)
    print numpy.shape(t_batch), numpy.shape(in_batch), numpy.shape(target_batch)

    config.STRICT_BATCH_SIZE = False
    t_batch, in_batch, target_batch = load_batch_for_nodes(dict_all_traces, 31, [], 0, True)
    print numpy.shape(t_batch), numpy.shape(in_batch), numpy.shape(target_batch)

    return


# if __name__ == '__main__':
#     test_sample()
