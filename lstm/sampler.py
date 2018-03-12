# coding:utf-8

import numpy
import copy

import utils

__all__ = [
    "Sampler",
    "GridSystem"
]


class GridSystem:
    def __init__(self, scale, base_xy=None):
        self.scale = scale
        self.base_xy = base_xy


class Sampler:
    def __init__(self, path=None, nodes=None, length=None, dimension_sample=None, length_sequence_input=None,
                 length_sequence_output=None, size_batch=None, strict_batch_size=None, keep_positive=True,
                 slot_trace=None):

        try:
            self.path = path

            # Note: Modifications of coordinates(panning, gridding, ...) must be applied to `self.dict_traces`,
            # and call `_update_traces_` manually afterwards to update `self.traces`.
            # Rather than apply modifications to `self.traces` directly
            self.dict_traces = {}  # {inode: {time: [x, y], ...}, ...}
            self.traces = None

            self.node_filter = nodes
            self.node_identifiers = {}  # by the order of `inode` in `self.dict_traces`
            self.num_node = None
            self.length_trace = 0  # counts of instants, minimum among all requested nodes
            self.length_limit = length if length else None
            self.motion_range = None

            self.entry = None  # the entry (in terms of length_trace) for batch loading

            self.dimension_sample = dimension_sample if dimension_sample is not None else utils.get_config(
                'dimension_sample')
            self.length_sequence_input = length_sequence_input \
                if length_sequence_input is not None else utils.get_config('length_sequence_input')
            self.length_sequence_output = length_sequence_output \
                if length_sequence_output is not None else utils.get_config('length_sequence_output')
            self.slot_trace = slot_trace if slot_trace is not None else utils.get_config('slot_trace')

            self.size_batch = size_batch if size_batch is not None else utils.get_config('size_batch')
            self.strict_batch_size = strict_batch_size if strict_batch_size is not None else utils.get_config(
                'strict_batch_size')
            self.grid_system = None  # for mapping to grids

            # read traces from files under given path
            if self.path is not None:
                self.path = utils.filer.validate_path_format(self.path)
                self._read_traces_from_path_()
                self._update_traces_()
                if keep_positive:
                    self.pan_to_positive()

            # create a empty sampler otherwise

        except:
            raise

    @staticmethod
    def empty_like(sampler):
        """
        Create an empty sampler for the same set of nodes as the given sampler.
        :param sampler: The sampler to copy the node container from.
        :return: The newly created empty sampler.
        """
        try:
            empty = Sampler(path=None, dimension_sample=sampler.dimension_sample,
                            length_sequence_input=sampler.length_sequence_input,
                            length_sequence_output=sampler.length_sequence_output, size_batch=sampler.size_batch,
                            strict_batch_size=sampler.strict_batch_size, keep_positive=False,
                            slot_trace=sampler.slot_trace)
            empty._copy_node_container_(sampler)
            return empty
        except:
            raise

    def _copy_node_container_(self, sampler):
        """
        Copy node identifiers and other necessary attributes from another sampler.
        :param sampler: The sampler to copy from.
        :return: None
        """
        try:
            utils.assertor.assert_type(sampler, Sampler)
            self.node_identifiers = sampler.node_identifiers
            self.num_node = sampler.num_node
            self.dict_traces = {inode: {} for inode in sampler.dict_traces}
            self._update_traces_()
        except:
            raise

    def _update_traces_(self):
        """
        Must call this method manually whenever `self.dict_traces` is changed.
        """
        self.traces = Sampler._dict_to_array_(self.dict_traces, self.dimension_sample, self.length_limit)
        self.num_node = int(self.traces.shape[0])
        self.length_trace = int(self.traces.shape[1])
        self.motion_range = Sampler._compute_range_(self.traces)

    def length(self):
        """
        Number of samples(lines).
        """
        return self.length_trace

    def empty(self):
        """
        Considered empty if having no samples.
        """
        return self.length() <= 0

    @staticmethod
    def _length_to_npair_(length, length_sequence_input, length_sequence_output):
        return length - (length_sequence_input + length_sequence_output - 1)

    @staticmethod
    def _npair_to_length_(npair, length_sequence_input, length_sequence_output):
        return npair + (length_sequence_input + length_sequence_output - 1)

    def npair(self):
        """
        Number of input-target pairs computed based on `length_sequence_input`, `length_sequence_output` & `length` .
        """
        n = Sampler._length_to_npair_(self.length(), self.length_sequence_input, self.length_sequence_output)
        return n if n > 0 else 0

    def reset_entry(self):
        if self.entry is not None:
            self.entry = 0

    @staticmethod
    def make_sample(sec, coors, dim=3):
        return [sec, coors[0], coors[1]] if dim >= 3 else coors

    @staticmethod
    def _read_triples_from_file_(filepath):
        """
        Read triples of (time, x, y) from given file, put in the returned dict by the key of time(/sec).
        Note: Format must be 'time x y'
        :param filepath: <string>
        :return: {time: [x, y]}
        """

        try:
            lines = utils.filer.read_lines(filepath)
            triples = {}  # {time: [x, y]}

            # Read a triple per line & save to dict
            for i in range(0, len(lines)):
                sec, x, y = [numpy.float32(x) for x in lines[i].split()][0:3]
                triples[sec] = numpy.array([x, y])

            return triples

        except:
            raise

    def _read_traces_from_path_(self):
        # todo add filename filter
        """
        Read traces from files under given path and put into `dict_traces`, filtered by `node_filter`.
        Note: Extension name validation is not included yet.
        Format of dict: dict{<string>: <list>[:num_line, :3]}
        :return: None
        """

        if not utils.filer.if_exists(self.path):
            raise IOError("Invalid path '%s'" % self.path)
        else:
            try:
                list_subdir = utils.filer.list_directory(self.path)
                dict_files = {utils.filer.split_extension(subdir)[0]: subdir for subdir in list_subdir if
                              utils.filer.is_file(utils.filer.format_subpath(self.path, subdir))}

                # Filter nodes

                nodes_filtered = utils.sorted_keys(dict_files)
                # by num_node
                if self.node_filter is None:
                    pass

                # by node identifiers
                elif isinstance(self.node_filter, list) \
                        and len(self.node_filter) > 0:
                    self.node_filter = ['%s' % node for node in self.node_filter]
                    for node in self.node_filter:
                        if node not in dict_files:
                            raise ValueError("Cannot find trace file for node '%s' under given path." % node)
                    nodes_filtered = self.node_filter

                else:
                    raise ValueError("Expect a node filter of <list>, "
                                     "while getting %s instead." % type(self.node_filter))

                dict_files_filtered = {node_id: dict_files[node_id] for node_id in nodes_filtered}
                node_identifiers = {inode: nodes_filtered[inode] for inode in xrange(len(nodes_filtered))}
                utils.xprint("Use nodes %s filtered by '%s'." % (utils.sorted_values(node_identifiers),
                                                                 self.node_filter), newline=True)

                # Actual reading from files

                dict_traces = {}
                for inode in xrange(len(nodes_filtered)):
                    node_id = nodes_filtered[inode]
                    filename = dict_files_filtered[node_id]
                    temp_triples = Sampler._read_triples_from_file_(self.path + filename)
                    dict_traces[inode] = temp_triples

            except:
                raise

        self.dict_traces = dict_traces
        self.node_identifiers = node_identifiers
        self.num_node = len(self.node_identifiers)

    @staticmethod
    def _dict_to_array_(dict_traces, dimension_sample, length=None):

        try:
            # 计算不同节点轨迹长度的最小值
            if length is None:
                length = min([len(trace) for trace in dict_traces.values()])
            # 以 length 为最后一行，对齐截断

            array_traces = numpy.array([
                                           [
                                               Sampler.make_sample(sec, dict_traces[inode][sec], dim=dimension_sample)
                                               for sec in utils.sorted_keys(dict_traces[inode])
                                               ][:length] if length > 0 else []
                                           for inode in utils.sorted_keys(dict_traces)
                                           ], dtype=numpy.float32)
            return array_traces

        except:
            raise

    def divide(self, trainset):
        """
        Divide this sampler into a train set & a test set, according to `trainset`.
        :param trainset: <float> among [0, 1] means ratio of `trainset / all`, <int> among (1, +) means size of trainset.
        :return: sampler_trainset, sampler_testset
        """
        try:
            trainset = int(trainset * self.npair()) if trainset < 1 else trainset
            sampler_trainset = Sampler.clip(self, indices=(0, trainset))
            sampler_testset = Sampler.clip(self, indices=(sampler_trainset.npair(), None))
            # if sampler_trainset.empty():
            #     sampler_trainset = None
            # if sampler_testset.empty():
            #     sampler_testset = None

            return sampler_trainset, sampler_testset
        except:
            raise

    @staticmethod
    def clip(a, indices=None):

        """
        Get a sampler clipped from an existed sampler by given indices (in terms of input-output sequence pair count).
        :param a: The sampler to be clipped from.
        :param indices: Slice range e.g. indices=(6, 10); indices=(0, None); indices=15 same as indices=(0,15)
                        Note: in terms of input-output sequence pair count
        :return: The clipped sampler.
        """
        try:
            utils.assertor.assert_type(a, Sampler)
            clipped = copy.deepcopy(a)
            if indices is None:
                return None
            elif isinstance(indices, list) \
                    or isinstance(indices, tuple):
                utils.assertor.assert_(len(indices) == 2, "Expect a list/tuple with length of 2, "
                                                          "while getting %d instead." % len(indices))
                ifrom, ito = indices[0], indices[1]
            elif isinstance(indices, int):
                ifrom, ito = 0, indices
            else:
                utils.assertor.assert_type(indices, [list, tuple, int])
                return None

            if ifrom < 0 \
                    or (ito is not None
                        and ito >= a.npair()):
                raise ValueError("Invalid indices (%d, %d). "
                                 "Index must be within [0, %d]."
                                 % (ifrom, ito, a.npair()))

            if ito is not None:
                ito = Sampler._npair_to_length_(ito, a.length_sequence_input, a.length_sequence_output)

            traces = a.traces
            traces = numpy.array([trace[ifrom:ito, :] for trace in traces], dtype=numpy.float32)
            clipped.traces = traces
            clipped.length_trace = numpy.shape(traces)[1]
            clipped.motion_range = Sampler._compute_range_(clipped.traces)
            return clipped

        except:
            raise

    @staticmethod
    def _compute_range_(array_traces):

        """

        :param array_traces:
        :return: [[x_min, y_min], [x_max, y_max]]
        """
        if not numpy.size(array_traces):
            return None

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

        stride = - self.motion_range[0, :]
        for inode in self.dict_traces:
            for sec in self.dict_traces[inode]:
                self.dict_traces[inode][sec][-2:] = self.dict_traces[inode][sec][-2:] + stride

        self._update_traces_()
        return stride

    def map_to_grid(self, grid_system=None):

        if grid_system is None:
            grid_system = GridSystem(utils.get_config('scale_grid'))
        if grid_system.base_xy is None:
            grid_system.base_xy = numpy.zeros((2,))
        for inode in self.dict_traces:
            for sec in self.dict_traces[inode]:
                self.dict_traces[inode][sec][-2:] = numpy.floor_divide(
                    (self.dict_traces[inode][sec][-2:] - grid_system.base_xy), grid_system.scale)

        self._update_traces_()
        self.grid_system = grid_system

    def _load_sample_(self):
        """
        指定起始位置，为单次训练/测试生成所有节点的输入序列（及目标序列）。
        注意：如果超出最大采样数，将返回 None，因此需要对返回值进行验证之后再使用
        :returns: (instants_input, sequences_input, instants_target, sequences_target)
        :format: instants_input: [length_sequence_input],
                 inputs: [num_node, length_sequence_input, dimension_sample],
                 instants_target: [length_sequence_output],
                 targets: [num_node, length_sequence_output, dimension_sample](None if not found)
        """

        try:
            if self.entry is None:
                self.entry = 0
            if self.entry >= self.length():
                return None, None, None, None

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

            # 如果超出采样数，返回 None, None, None, None
            if end_line_input > self.length():
                return None, None, None, None

            instants_input = instants[begin_line_input: end_line_input]
            sequences_input = array_traces[:, begin_line_input: end_line_input, :]

            instants_target = instants_input + self.slot_trace * self.length_sequence_input
            instants_target = instants_target[:self.length_sequence_output]

            # sequences_target is None if not found
            begin_line_target = end_line_input
            end_line_target = begin_line_target + self.length_sequence_output
            if end_line_target > self.length():
                sequences_target = None
            else:
                sequences_target = array_traces[:, begin_line_target: end_line_target, :]

            return instants_input, sequences_input, instants_target, sequences_target

        except:
            raise

    def load_batch(self, size_batch=None):
        """
        为单批次的训练/测试生成所有节点的时刻、输入、目标序列。
        注意：如果超出最大采样数，将返回 4 个 None，因此需要对返回值进行验证之后再使用
        :returns: (instants_input, sequences_input, instants_target, sequences_target)
        :format: instants_input: [size_batch, length_sequence_input],
                 sequences_inputs: [num_node, size_batch, length_sequence_input, dimension_sample],
                 instants_target: [size_batch, length_sequence_output],
                 sequences_targets: [num_node, size_batch, length_sequence_output, dimension_sample] (None if not found)
        """

        if size_batch is None:
            size_batch = self.size_batch
        try:
            batch_instants_input = numpy.zeros((0, self.length_sequence_input))
            batch_instants_target = numpy.zeros((0, self.length_sequence_output))
            batch_input = numpy.zeros((0, 0, 0, 0))
            batch_target = numpy.zeros((0, 0, 0, 0))
            for i in range(size_batch):
                sample_instant_input, sample_input, sample_instant_target, sample_target = self._load_sample_()
                if sample_input is not None:
                    self.entry += 1
                    shape_input = sample_input.shape

                    batch_instants_input = numpy.append(batch_instants_input,
                                                        numpy.expand_dims(sample_instant_input, axis=0), axis=0)
                    batch_instants_target = numpy.append(batch_instants_target,
                                                         numpy.expand_dims(sample_instant_target, axis=0), axis=0)
                    batch_input = numpy.resize(batch_input, (shape_input[0], i + 1, shape_input[1], shape_input[2]))

                    for inode in range(shape_input[0]):
                        batch_input[inode, i] = sample_input[inode]

                    if sample_target is not None:
                        shape_target = sample_target.shape
                        batch_target = numpy.resize(batch_target,
                                                    (shape_target[0], i + 1, shape_target[1], shape_target[2]))
                        for inode in range(shape_target[0]):
                            batch_target[inode, i] = sample_target[inode]
                    else:
                        # return None if any one in this batch is not found
                        batch_target = None

                elif len(batch_input) == 0:
                    return None, None, None, None
                elif self.strict_batch_size:
                    utils.warn("Sampler.load_batch: "
                               "An insufficient batch of %s samples is discarded."
                               % batch_input.shape[1])
                    return None, None, None, None
                elif not self.strict_batch_size:
                    utils.warn("Sampler.load_batch: "
                               "Insufficient batch. Only %s samples are left."
                               % batch_input.shape[1])
                    break
                else:
                    utils.assertor.assert_unreachable()
            return batch_instants_input, batch_input, batch_instants_target, batch_target

        except:
            raise

    def retrieve_by_instants(self, instants):
        """
        Return <dict> of trace samples for all the given instants & all the nodes.
        :param instants: array of all the unique instants to retrieve by.
        :return: <dict>{second: {node: coordinates, ...}, ...} (maybe empty, check before use)
        """
        try:
            result = {}
            if not utils.assertor.assert_type(instants, [numpy.ndarray, list]) or not numpy.size(instants):
                return result

            for inode in self.node_identifiers:
                for sec in instants:
                    if sec in self.dict_traces[inode]:
                        if sec not in result:
                            result[sec] = {}
                        result[sec][inode] = self.dict_traces[inode][sec]

            return result
        except:
            raise

    @staticmethod
    def _update_batch_input_(batch_instants, batch_inputs, indicator_updates, dict_updates):

        """
        Update certain samples into given batch according to the instants entries. Return the updated batch.
        :param batch_instants: `batch_instants_input` returned by `load_batch`,
                               [size_batch, length_sequence_input]
        :param batch_inputs: `batch_input` returned by `load_batch`,
                             [num_node, size_batch, length_sequence_input, dimension_sample]
        :param indicator_updates: indicator matrix, 1 for the elements to be updated, 0 for otherwise.
                                  Should be the same dimension as `batch_instants`.
        :param dict_updates: <dict> returned by `retrieve_by_instants`
        :return: [num_node, size_batch, length_sequence_input, dimension_sample]
        """
        try:
            if indicator_updates is None \
                    or not numpy.any(indicator_updates) \
                    or not dict_updates:
                return batch_inputs

            num_node, size_batch, len_seq, _ = numpy.shape(batch_inputs)
            for ibatch in xrange(size_batch):
                for iseq in xrange(len_seq):
                    if indicator_updates[ibatch, iseq]:
                        for inode in xrange(num_node):
                            sec = batch_instants[ibatch, iseq]
                            if sec in dict_updates \
                                    and inode in dict_updates[sec]:
                                batch_inputs[inode, ibatch, iseq, -2:] = dict_updates[sec][inode]
                    else:
                        pass

            return batch_inputs

        except:
            raise

    @staticmethod
    def make_unreliable_input(inputs, predictions, instants, unreliability):
        """
        When unreliable input is enabled, update certain elements in `inputs` with `predictions`.
        :param inputs: Original & reliable inputs of a single batch, returned by `load_batch`.
        :param predictions: The <Sampler> that contains prediction results from previous training.
        :param instants: Instants for `inputs`, returned by `load_batch`.
        :param unreliability: The degree of unreliability.
                              a) `False` means disabled,
                              b) <int> among [1, length_sequence_input) means how many of most recent predictions
                                 (instants) is allowed to used.
        :return:
        """
        try:
            if not unreliability \
                    or (utils.assertor.assert_type(predictions, Sampler)
                        and predictions.empty()):
                return inputs

            instants_update = instants[:, -unreliability:]
            instants_update = numpy.unique(instants_update)
            dict_updates = predictions.retrieve_by_instants(instants_update)

            indicator_updates = numpy.zeros_like(instants)
            indicator_updates[:, -unreliability:] = 1
            inputs = Sampler._update_batch_input_(instants, inputs, indicator_updates, dict_updates)
            return inputs
        except:
            raise

    def save_batch_output(self, instants_output, outputs):
        """

        :param instants_output: `batch_instants_input` returned by `load_batch`,
                               [size_batch, length_sequence_output]
        :param outputs: `batch_input` returned by `load_batch`,
                             [num_node, size_batch, length_sequence_output, dimension_sample]
        :return: None
        """
        try:
            num_node, size_batch, len_seq, dim = numpy.shape(outputs)
            instants_output = numpy.ndarray.flatten(instants_output)
            outputs = numpy.reshape(outputs, (num_node, size_batch * len_seq, dim))

            for inode in xrange(num_node):
                for ibatch in xrange(len(instants_output)):
                    sec, coors = instants_output[ibatch], outputs[inode, ibatch, -2:]
                    self.dict_traces[inode][sec] = coors

            self._update_traces_()

        except:
            raise

    @staticmethod
    def test():

        try:
            utils.xprint('Testing Sampler... ', newline=True)

            def test_init():
                dict_triples = Sampler._read_triples_from_file_('res/trace/NCSU/2.trace')

                sampler = Sampler(utils.get_config('path_trace'), nodes=3, length=18)
                sampler = Sampler(utils.get_config('path_trace'), nodes=['2'])
                sampler = Sampler(utils.get_config('path_trace'))

                sampler_empty = Sampler.empty_like(sampler)

            test_init()

            def test_grid_and_pan():
                sampler = Sampler(utils.get_config('path_trace'), nodes=3, keep_positive=False)
                sampler.map_to_grid(GridSystem(100))
                sampler.pan_to_positive()
                sampler.pan_to_positive()

            test_grid_and_pan()

            def test_clip_and_divide():
                sampler = Sampler(utils.get_config('path_trace'), nodes=3)
                twenty = Sampler.clip(sampler, indices=20)
                try:
                    invalid = Sampler.clip(twenty, indices=(-1, 20))
                except Exception, e:
                    assert isinstance(e, ValueError)
                five = Sampler.clip(twenty, indices=(5, None))

                trainset, testset = twenty.divide(0.25)

            test_clip_and_divide()

            def test_load():
                sampler = Sampler(utils.get_config('path_trace'), nodes=3, size_batch=5)
                instants_inputs, inputs, instants_targets, targets = sampler.load_batch()

                sampler.reset_entry()
                while True:
                    # 1 batch for each node
                    instants_inputs, inputs, instants_targets, targets = sampler.load_batch()
                    peek_entry = sampler.entry
                    if inputs is None:
                        break

            test_load()

            def test_loose_batch_size():
                sampler = Sampler(utils.get_config('path_trace'), nodes=3, size_batch=5)
                sampler.strict_batch_size = False
                while True:
                    # 1 batch for each node
                    instants_inputs, inputs, instants_targets, targets = sampler.load_batch()
                    peek_entry = sampler.entry
                    if inputs is None:
                        break

            test_loose_batch_size()

            def test_unreliable_inputs():
                sampler = Sampler(utils.get_config('path_trace'), nodes=['2', '3'],
                                  length_sequence_output=2, size_batch=3)
                sampler_predictions = Sampler()
                sampler_predictions._copy_node_container_(sampler)

                instants_input, inputs, instants_target, targets = sampler.load_batch()

                targets *= 0
                sampler_predictions.save_batch_output(instants_target, targets)

                try:
                    sampler_predictions.retrieve_by_instants(30)
                except ValueError, e:
                    pass
                sampler_predictions.retrieve_by_instants([30])

                # unreliable_input = False
                unreliable_input = 1  # 1-HOP
                inputs = Sampler.make_unreliable_input(inputs, sampler_predictions, instants_input, unreliable_input)

            test_unreliable_inputs()

            utils.xprint('...Fine', newline=True)

        except:
            raise
