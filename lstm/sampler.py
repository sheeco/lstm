# coding:utf-8
import numpy
import copy

import config
import utils
import filer

__all__ = [
    "Sampler",
    "GridSystem"
    ]


class GridSystem:

    def __init__(self, grain, base_xy=None):
        self.grain = grain
        self.base_xy = base_xy


class Sampler:

    def __init__(self, path=config.PATH_TRACE_FILES, nodes=None, length=None, dimension_sample=config.DIMENSION_SAMPLE, length_sequence_input=config.LENGTH_SEQUENCE_INPUT,
                 length_sequence_output=config.LENGTH_SEQUENCE_OUTPUT, size_batch=config.SIZE_BATCH, strict_batch_size=config.STRICT_BATCH_SIZE, keep_positive=True):

        try:
            self.path = path
            dict_traces = Sampler.__read_traces_from_path__(path)
            dict_traces = Sampler.__filter__(dict_traces, nodes)
            self.node_identifiers = dict_traces.keys()
            self.node_filter = nodes
            self.traces = Sampler.__dict_to_array__(dict_traces, length)
            self.num_node = int(self.traces.shape[0])
            self.length = int(self.traces.shape[1])
            self.motion_range = Sampler.__compute_range__(self.traces)
            self.grid_system = None
            self.entry = 0

            self.dimension_sample = dimension_sample
            self.length_sequence_input = length_sequence_input
            self.length_sequence_output = length_sequence_output
            self.size_batch = size_batch
            self.strict_batch_size = strict_batch_size
            if keep_positive:
                self.pan_to_positive()

        except:
            raise

    def reset_entry(self):

        self.entry = 0

    @staticmethod
    def __read_triples_from_file__(filename):
        """
        从单个文件读入 (time, x, y) 的三元组序列，放入返回 list。
        注意：暂无格式检查，文件中每一行必须对应一次定位采样，格式为 'time x y'
        :param filename: string 文件名
        :return: list[:行数, :3]
        """

        try:
            lines = filer.read_lines(filename)
            triples = []  # (time, x, y) 的三元组列表

            # 从每一行读入三个数值并存入列表中的一行
            for i in range(0, len(lines)):
                triples.append([numpy.float32(x) for x in lines[i].split()][0:3])

            # n_triples = len(triples)

            return numpy.array(triples, dtype=numpy.float32)

        except:
            raise

    @staticmethod
    def __read_traces_from_path__(path):
        # todo add filename filter
        """
        对给定目录下的所有文件，读取轨迹序列，放入返回 dict。
        注意：暂无扩展名检查
        :param path: string 指定路径
        :return: dict{轨迹文件名 string: 采样三元组 list[:行数, :3]}
        """

        dict_traces = {}

        if not filer.if_exists(path):
            raise IOError('read_traces_from_path @ sample: \n\tInvalid Path: %s' % path)
        else:
            try:
                list_subdir = filer.list_directory(path)
                list_files = [subdir for subdir in list_subdir if filer.is_file(path + subdir)]

                for filename in list_files:
                    node_identifier, _ = filer.split_extension(filename)
                    temp_trace = Sampler.__read_triples_from_file__(path + filename)
                    # if node_name.isdigit():
                    #     traces[int(node_name)] = temp_trace
                    # else:
                    #     traces[node_name] = temp_trace
                    dict_traces[node_identifier] = temp_trace

            except:
                raise

        return dict_traces

    @staticmethod
    def __filter__(dict_traces, node_filter):

        try:
            if node_filter is None:
                return dict_traces
            # 指定节点个数
            if isinstance(node_filter, int):
                if 0 < node_filter < len(dict_traces):
                    nodes_requested = dict_traces.keys()[:node_filter]
                else:
                    if node_filter > len(dict_traces):
                        utils.warn("__filter__ @ Sampler: Cannot find enough nodes in the given path.")
                    return dict_traces

            # 指定 node identifiers
            elif isinstance(node_filter, list) and len(node_filter) > 0:
                nodes_requested = node_filter
            else:
                return dict_traces

            dict_traces_requested = {node_id: dict_traces[node_id] for node_id in nodes_requested}
            return dict_traces_requested

        except:
            raise

    @staticmethod
    def __dict_to_array__(dict_traces, length=None):

        try:
            # 计算不同节点轨迹长度的最小值
            if length is None:
                length = min([len(trace) for trace in dict_traces.values()])
            # 以 length 为最后一行，对齐截断
            # 并将选中 dict 中的 value 转换成 array，顺序由给定的 node_identifiers 决定
            array_traces = numpy.array([trace[:length, :] for trace in dict_traces.values()],
                                           dtype=numpy.float32)
            return array_traces

        except:
            raise

    @staticmethod
    def clip(a, indices=None):

        """

        :param a: The sampler to be clipped
        :param indices: Slice range e.g. indices=(6, 10); or n for (0, n) e.g. indices=5
        :return: A clipped Sampler copied from `a`
        """
        try:
            utils.assert_type(a, Sampler)
            out = copy.deepcopy(a)
            if indices is None:
                return None
            elif isinstance(indices, list) or isinstance(indices, tuple):
                assert len(indices) == 2
                ifrom, ito = indices[0], indices[1]
            elif isinstance(indices, int):
                ifrom, ito = 0, indices
            else:
                utils.assert_type(indices, [list, tuple, int])
                return None
            traces = a.traces
            traces = numpy.array([trace[ifrom:ito, :] for trace in traces], dtype=numpy.float32)
            out.traces = traces
            out.length = int(out.traces.shape[1])
            out.motion_range = Sampler.__compute_range__(out.traces)
            return out

        except:
            raise

    @staticmethod
    def __compute_range__(array_traces):

        """

        :param array_traces:
        :return: [[x_min, y_min], [x_max, y_max]]
        """
        motion_range = numpy.zeros((2, 2))
        samples = numpy.reshape(array_traces, newshape=(-1, array_traces.shape[-1]))
        motion_range[0, 0] = numpy.min(samples[:, 1])
        motion_range[0, 1] = numpy.min(samples[:, 2])
        motion_range[1, 0] = numpy.max(samples[:, 1])
        motion_range[1, 1] = numpy.max(samples[:, 2])

        motion_range[0, :] = numpy.floor(motion_range[0, :])
        motion_range[1, :] = numpy.ceil(motion_range[1, :])

        return motion_range

    def pan_to_positive(self):

        if numpy.min(self.motion_range) >= 0.:
            return self.traces

        traces = self.traces
        stride = - self.motion_range[0, :]
        for trace in traces:
            for triple in trace:
                triple[1:3] = triple[1:3] + stride

        self.motion_range += stride
        self.traces = traces
        return self.traces

    def map_to_grid(self, grid_system=GridSystem(config.GRAIN_GRID)):

        traces = self.traces
        if grid_system.base_xy is None:
            grid_system.base_xy = numpy.floor_divide(self.motion_range[0, :], grid_system.grain) * grid_system.grain
        for trace in traces:
            for triple in trace:
                triple[1:3] = numpy.floor_divide((triple[1:3] - grid_system.base_xy), grid_system.grain)

        self.motion_range = Sampler.__compute_range__(traces)
        self.traces = traces
        self.grid_system = grid_system
        return self.traces

    def __load_sample__(self, with_target=True):
        """
        指定起始位置，为单次训练/测试生成所有节点的输入序列（及目标序列）。
        注意：如果超出最大采样数，将返回 None，因此需要对返回值进行验证之后再使用
        :format: [:LENGTH_SEQUENCE_INPUT], [:所选节点数, :LENGTH_SEQUENCE_INPUT, :DIMENSION_SAMPLE], [:所选节点数, :LENGTH_SEQUENCE_OUTPUT， :DIMENSION_SAMPLE]
        """

        try:
            if self.entry >= self.length:
                return None, None, None

            array_traces = self.traces
            # 提取时序
            instants = array_traces[:, :, :1]
            instants = instants[0]
            instants = instants.sum(1)

            if self.dimension_sample == 2:
                # 不使用 time 作为输入元组的一部分，删除第 0 列
                array_traces = array_traces[:, :, 1:]

            begin_line_input = self.entry
            end_line_input = begin_line_input + self.length_sequence_input
            begin_line_target = end_line_input
            end_line_target = begin_line_target + self.length_sequence_output

            # 如果超出采样数，返回 None, None, None
            if end_line_input >= self.length or end_line_target > self.length:
                return None, None, None
            else:
                self.entry = end_line_input

            sequences_instants = instants[begin_line_input: end_line_input]
            sequences_input = array_traces[:, begin_line_input: end_line_input, :]

            sequences_target = array_traces[:, begin_line_target: end_line_target, :] if with_target else None

            return sequences_instants, sequences_input, sequences_target

        except:
            raise


    def load_batch(self, size_batch=None, with_target=True):
        """
        为单批次的训练/测试生成所有节点的时刻、输入、目标序列。
        注意：如果超出最大采样数，将返回 3 个 None，因此需要对返回值进行验证之后再使用
        :format: [size_batch, LENGTH_SEQUENCE_INPUT], [N_NODES, size_batch, LENGTH_SEQUENCE_INPUT, DIMENSION_SAMPLE], [N_NODES, size_batch, LENGTH_SEQUENCE_OUTPUT, DIMENSION_SAMPLE]
        """
        # todo add boolean arg redundant_batch

        if size_batch is None:
            size_batch = self.size_batch
        try:
            batch_instants = numpy.zeros((0, 0))
            batch_input = numpy.zeros((0, 0, 0, 0))
            batch_target = numpy.zeros((0, 0, 0, 0))
            for i in range(size_batch):
                sample_instants, sample_input, sample_target = self.__load_sample__(with_target)
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
                        batch_target = numpy.resize(batch_target,
                                                    (shape_target[0], i + 1, shape_target[1], shape_target[2]))
                        for inode in range(shape_target[0]):
                            batch_target[inode, i] = sample_target[inode]

                elif len(batch_input) == 0:
                    return None, None, None
                elif self.strict_batch_size:
                    utils.warn("load_batch @ Sampler: An insufficient batch of %s samples is discarded." % batch_input.shape[1])
                    return None, None, None
                elif not self.strict_batch_size:
                    utils.warn("load_batch @ Sampler: Insufficient batch. Only  %s  samples are left." % batch_input.shape[1])
                    break
                else:
                    utils.assert_unreachable()
            if not with_target:
                batch_target = None
            return batch_instants, batch_input, batch_target

        except:
            raise

    @staticmethod
    def test():

        try:
            utils.xprint('Testing Sampler...', level=1)

            demo_list_triples = Sampler.__read_triples_from_file__('res/trace/2.trace')
            # utils.xprint(numpy.shape(demo_list_triples), level=1, newline=True)

            sampler = Sampler(config.PATH_TRACE_FILES, nodes=1)
            sampler = Sampler(config.PATH_TRACE_FILES, length=18)
            sampler = Sampler(config.PATH_TRACE_FILES, nodes=['2'])

            sampler = Sampler(config.PATH_TRACE_FILES)
            sampler.pan_to_positive()
            sampler.pan_to_positive()
            sampler.map_to_grid(GridSystem(100))

            clipped = Sampler.clip(sampler, indices=20)

            instants, inputs, targets = sampler.load_batch(with_target=False)
            # utils.xprint([to_check.shape if to_check is not None else 'None' 
            # for to_check in (instants, inputs, targets)], level=1, newline=True)

            sampler.strict_batch_size = False
            while True:
                # 1 batch for each node
                instants, inputs, targets = sampler.load_batch(with_target=True)
                # utils.xprint([to_check.shape if to_check is not None else 'None' 
                # for to_check in (instants, inputs, targets)], level=1, newline=True)
                check_entry = sampler.entry
                if inputs is None:
                    break

            utils.xprint('Fine', level=1, newline=True)
            return True

        except:
            raise
