# coding:utf-8

import numpy

import theano
import theano.tensor as T
import lasagne as L

import utils
from sampler import *

__all__ = [
    'SocialLSTM'
]


class SocialLSTM:

    # todo add None
    # todo add the sharing of social tensor H maybe
    SHARE_SCHEMES = ['parameter',
                     'input',
                     'olstm',
                     'none']

    TRAIN_SCHEMES = ['rmsprop',
                     'adagrad',
                     'momentum',
                     'nesterov']

    DECODE_SCHEMES = ['binorm',
                      'euclidean']

    def __init__(self, node_identifiers, motion_range, samples=None, targets=None,
                 share_scheme=None, decode_scheme=None, train_scheme=None,
                 scale_pool=None, range_pool=None, hit_range=None,
                 dimension_sample=None, length_sequence_input=None, length_sequence_output=None, size_batch=None,
                 dimension_embed_layer=None, dimension_hidden_layer=None,
                 learning_rate=None, rho=None, epsilon=None, momentum=None, grad_clip=None, num_epoch=None,
                 adaptive_learning_rate=None, adaptive_grad_clip=None):
        try:
            if __debug__:
                theano.config.exception_verbosity = 'high'
                theano.config.optimizer = 'fast_compile'

            # Node related variables

            self.node_identifiers = node_identifiers
            self.num_node = len(self.node_identifiers)
            self.motion_range = motion_range

            # Variables defining the network
            self.share_scheme = share_scheme if share_scheme is not None else utils.get_config('share_scheme')
            if self.share_scheme not in SocialLSTM.SHARE_SCHEMES:
                raise ValueError(
                    "Unknown sharing scheme '%s'. Must be among %s." % (self.share_scheme, SocialLSTM.SHARE_SCHEMES))

            self.dimension_sample = dimension_sample \
                if dimension_sample is not None else utils.get_config('dimension_sample')
            self.length_sequence_input = length_sequence_input \
                if length_sequence_input is not None else utils.get_config('length_sequence_input')
            self.length_sequence_output = length_sequence_output \
                if length_sequence_output is not None else utils.get_config('length_sequence_output')
            self.size_batch = size_batch \
                if size_batch is not None else utils.get_config('size_batch')
            self.dimension_embed_layer = dimension_embed_layer \
                if dimension_embed_layer is not None else utils.get_config('dimension_embed_layer')
            dimension_hidden_layer = dimension_hidden_layer \
                if dimension_hidden_layer is not None else utils.get_config('dimension_hidden_layer')
            utils.assertor.assert_type(dimension_hidden_layer, tuple)
            all(utils.assertor.assert_type(x, int) for x in dimension_hidden_layer)

            if len(dimension_hidden_layer) == 1:
                dimension_hidden_layer = (1,) + dimension_hidden_layer
            elif len(dimension_hidden_layer) == 2:
                pass
            else:
                raise ValueError("Expect len: 1~2 while getting %d instead.",
                                 len(dimension_hidden_layer))
            self.num_hidden_layer = dimension_hidden_layer[0]
            self.dimension_hidden_layer = dimension_hidden_layer[1]
            self.grad_clip = grad_clip if grad_clip is not None else utils.get_config('grad_clip')
            self.decode_scheme = decode_scheme if decode_scheme is not None else utils.get_config('decode_scheme')

            if self.decode_scheme not in SocialLSTM.DECODE_SCHEMES:
                raise ValueError("Unknown loss scheme '%s'. Must choose among %s." % (self.decode_scheme, SocialLSTM.DECODE_SCHEMES))

            self.train_scheme = train_scheme if train_scheme is not None else utils.get_config('train_scheme')
            if self.train_scheme not in SocialLSTM.TRAIN_SCHEMES:
                raise ValueError(
                    "Unknown training scheme '%s'. Must be among %s." % (self.train_scheme, SocialLSTM.TRAIN_SCHEMES))

            # neighborhood definition for social pooling
            self.scale_pool = scale_pool \
                if scale_pool is not None else utils.get_config('scale_pool')
            self.range_pool = range_pool \
                if range_pool is not None else utils.get_config('range_pool')

            # for hitrate computation
            self.hit_range = hit_range \
                if hit_range is not None else utils.get_config('hit_range')

            self.learning_rate = learning_rate \
                if learning_rate is not None else utils.get_config('learning_rate')

            if self.train_scheme in ('rmsprop', 'adagrad'):
                self.epsilon = epsilon if epsilon is not None else utils.get_config('epsilon')
            if self.train_scheme == 'rmsprop':
                self.rho = rho if rho is not None else utils.get_config('rho')
            if self.train_scheme in ('momentum', 'nesterov'):
                self.momentum = momentum if momentum is not None else utils.get_config('momentum')

            # Training related variables

            self.num_epoch = num_epoch if num_epoch is not None else utils.get_config('num_epoch')

            self.adaptive_learning_rate = adaptive_learning_rate if adaptive_learning_rate is not None \
                else utils.get_config('adaptive_learning_rate')
            self.adaptive_grad_clip = adaptive_grad_clip if adaptive_grad_clip is not None \
                else utils.get_config('adaptive_grad_clip')

            # Theano tensor variables (symbolic)

            if samples is None:
                samples = T.tensor4("samples", dtype='float32')
            if targets is None:
                targets = T.tensor4("targets", dtype='float32')

            # tensors, should take single batch of samples as real value
            self.samples = samples
            self.targets = targets

            # list of tensors, all the necessary input tensors for network
            # , only contains sample inputs by default
            self.network_inputs = [self.samples]
            self.network_outputs = None  # tensor, output of the entire network

            self.embed = None  # tensor, output of embedding layer
            self.hids_all = []  # 2d list of tensors, outputs of all the hidden layers of all the nodes
            self.hid_last = None  # tensors, output of the concatenated last hidden layers of all the nodes
            self.network = None  # Lasagne layer object, containing the network structure
            self.params_all = None  # dict of tensors, all the parameters
            self.params_trainable = None  # dict of tensors, trainable parameters
            self.updates = None  # dict of tensors, returned from predefined training functions in Lasagne.updates

            # # reserved for social tensor sharing
            # self.prev_hids = None
            # self.social_tensors = None

            # Real values stored for printing / debugging

            self.entry_epoch = 0
            self.entry_batch = 0
            self.stop = False  # to mark for manual stop

            self.loss = None  # numpy float, loss computed for single batch
            self.predictions = None  # numpy ndarray, predictions of single batch
            self.deviations = None  # numpy ndarray, euclidean distances between predictions & targets
            self.probabilities = None  # numpy ndarray, binorm probabilities before computing into NNL

            self.param_names = []  # list of str, names of all the parameters
            self.current_param_values = None  # list of ndarrays, stored for debugging or exporting
            self.initial_param_values = None  # ..., stored for possible parameter restoration
            self.best_param_values = {  # stored for possible parameter export
                'epoch': None,  # after n^th epoch
                'record': None,  # best record (hitrate) stored for comparison
                'value': None,  # actual param values
                'path': None}  # path of pickled file

            # Theano function objects

            self.func_predict = None
            self.func_compare = None
            self.func_train = None

            self.peek_outputs = None
            self.peek_params = None

            """
            Initialization Definitions
            """

            self.w_e = L.init.Uniform(std=0.005, mean=(1. / self.dimension_sample))
            self.b_e = L.init.Constant(0.)
            self.f_e = None

            # _dim_a = (2 * self.range_pool) ** 2 * self.dimension_hidden_layer
            # self.w_a = L.init.Uniform(std=0.005, mean=1. / _dim_a)
            # self.b_a = L.init.Constant(0.)
            # self.f_a = None

            _dim_o = (2 * self.range_pool) ** 2 * 1
            self.w_o = L.init.Uniform(std=0.005, mean=(1. / _dim_o))
            self.b_o = L.init.Constant(0.)
            self.f_o = None

            self.w_lstm_in = L.init.Uniform(std=0.005, mean=(1. / self.dimension_embed_layer))
            self.w_lstm_hid = L.init.Uniform(std=0.005, mean=(1. / self.dimension_hidden_layer))
            self.w_lstm_cell = L.init.Uniform(std=0.005, mean=(1. / self.dimension_hidden_layer))
            self.b_lstm = L.init.Constant(0.)
            self.f_lstm_hid = L.nonlinearities.rectify
            self.f_lstm_cell = L.nonlinearities.rectify
            self.init_lstm_hid = L.init.Constant(0.)
            self.init_lstm_cell = L.init.Constant(0.)

            self.w_means = L.init.Uniform(std=0.005, mean=(1. / self.dimension_hidden_layer))
            self.b_means = L.init.Constant(0.)
            self.f_means = None

            self.scaled_sigma = True

            self.w_deviations = L.init.Uniform(std=0.1, mean=(100. / self.dimension_hidden_layer / self.num_node))
            self.b_deviations = L.init.Constant(0.)
            if self.scaled_sigma:
                self.f_deviations = L.nonlinearities.sigmoid
            else:
                self.f_deviations = L.nonlinearities.softplus

            self.scaled_correlation = True

            # self.w_correlation = L.init.Uniform(std=0.0005, mean=0.)
            self.w_correlation = L.init.Uniform(std=0., mean=0.)
            self.b_correlation = L.init.Constant(0.)
            if self.scaled_correlation:
                self.f_correlation = SocialLSTM.scaled_tanh
            else:
                self.f_correlation = SocialLSTM.safe_tanh

            # Prepare for logging

            self.logger = utils.get_sublogger()

            _columns = ['epoch', 'batch', 'sample', 'instant'] + self.node_identifiers
            self.logger.register("train-sample", columns=_columns)
            self.logger.register("train-batch", columns=['epoch', 'batch', 'loss',
                                                         'mean-deviation', 'min-deviation', 'max-deviation'])
            self.logger.register("train-epoch", columns=["epoch", "mean-loss",
                                                         "mean-deviation", "min-deviation", "max-deviation"])
            self.logger.register("train-hitrate", columns=["epoch", "hitrate"])

            self.logger.register("test-sample", columns=_columns)
            self.logger.register("test-batch", columns=['epoch', 'batch', 'loss',
                                                        'mean-deviation', 'min-deviation', 'max-deviation'])
            self.logger.register("test-epoch", columns=["epoch", "mean-loss",
                                                        "mean-deviation", "min-deviation", "max-deviation"])
            self.logger.register("test-hitrate", columns=["epoch", "hitrate"])

        except:
            raise

    @staticmethod
    def bivar_norm(x1, x2, mu1, mu2, sigma1, sigma2, rho):
        """
        pdf of bivariate norm
        """
        try:
            part1 = (x1 - mu1) ** 2 / sigma1 ** 2
            part2 = - 2. * rho * (x1 - mu1) * (x2 - mu2) / sigma1 * sigma2
            part3 = (x2 - mu2) ** 2 / sigma2 ** 2
            z = part1 + part2 + part3

            cof = 1. / (2. * numpy.pi * sigma1 * sigma2 * T.sqrt(1 - rho ** 2))
            return cof * T.exp(-z / (2. * (1 - rho ** 2)))

        except:
            raise

    @staticmethod
    def clip(x, beta=.9):

        try:
            beta = T.as_tensor_variable(beta)
            return T.clip(x, -beta, beta)
        except:
            raise

    @staticmethod
    def scale(x, beta=.9):

        try:
            beta = T.constant(beta, name='stds-scale-ratio')
            return T.mul(beta, x)
        except:
            raise

    @staticmethod
    def scaled_tanh(x, beta=1.e-8):

        try:
            y = T.tanh(x)
            return SocialLSTM.scale(y, beta)
            # return T.clip(y, -beta, beta)
        except:
            raise

    @staticmethod
    def safe_tanh(x, beta=.9):

        try:
            y = T.tanh(x)
            return SocialLSTM.scale(y, beta)
            # return T.clip(y, -beta, beta)
        except:
            raise

    def compute_hitrate(self, deviations, nbin=None, formatted=False):
        """

        :param deviations: numpy array
        :param nbin: Only returns n first bins of hitrate histogram.
        :param formatted: Return formatted string otherwise raw numbers.
        :return: e.g. "50: 61.2%, 100: 20.7%, ..." if `formatted` else [(50, 61.2), (100, 20.7), ...]
        """
        try:
            step = self.hit_range
            deviations = numpy.reshape(deviations, newshape=(-1, deviations.shape[-1]))
            ceil = int(numpy.ceil(numpy.max(deviations) / step))

            # clip to avoid unnecessary computation
            MAX_CEIL = 10
            if ceil > MAX_CEIL:
                ceil = MAX_CEIL
            if nbin is not None \
                    and ceil < nbin:
                ceil = nbin
            ceil *= step

            edges = numpy.arange(start=0, stop=ceil + step, step=step)
            hist, _ = numpy.histogram(deviations, bins=edges)
            hist = numpy.true_divide(hist, numpy.size(deviations))

            if nbin is not None \
                    and nbin > 0:
                hist = hist[:nbin]

            for ibin in range(len(hist) - 1, 0, -1):
                # sum from x_0 to x_i
                hist[ibin] = numpy.sum(hist[:ibin + 1])

            if formatted:
                hitrates = ''
                for ibin in xrange(len(hist)):
                    if ibin > 0:
                        hitrates += ', '
                    hitrange = '%d' % edges[ibin + 1]
                    hitrate = '%.1f' % (hist[ibin] * 100) + '%'
                    hitrates += '%s: %s' % (hitrange, hitrate)
            else:
                hitrates = []
                for ibin in xrange(len(hist)):
                    hitrange = edges[ibin + 1]
                    hitrate = round(hist[ibin] * 100, ndigits=1)
                    hitrates += [(hitrange, hitrate)]

            return hitrates

        except:
            raise

    """
    input: samples(node, batch, seq, dim_sample)
    output: final occupancy_map: (node, batch, seq, m, n, 1)
    """
    def compute_occupancy_map(self, samples):
        try:

            # samples = T.tensor4('samples', dtype='float32')
            # prev_hids = T.tensor4('prev_hids', dtype='float32')

            shape_samples = samples.shape
            num_node = shape_samples[0]
            size_batch = shape_samples[1]
            len_sequence = shape_samples[2]

            # ones = T.constant(T.ones(shape=(num_node, size_batch, len_sequence, 1)), name="ones-hids")
            ones = T.ones(shape=(num_node, size_batch, len_sequence, 1))
            theano.gradient.zero_grad(ones)

            return self.compute_social_tensors(samples, ones)

        except:
            raise

    """
    input: samples(node, batch, seq, dim_sample), prev_hids (node, batch, seq, dim_hid)
    output: final social tensor: (node, batch, seq, m, n, dim)
    """
    def compute_social_tensors(self, samples, prev_hids):
        try:

            # samples = T.tensor4('samples', dtype='float32')
            # prev_hids = T.tensor4('prev_hids', dtype='float32')

            # samples = self.samples
            # prev_hids = self.prev_hids
            relu_hids = T.nnet.relu(prev_hids)

            shape_samples = samples.shape
            num_node = shape_samples[0]
            size_batch = shape_samples[1]
            len_sequence = shape_samples[2]

            def make_indice(num_node_var, size_batch_var, len_sequence_var):
                shape_indice = (num_node_var * size_batch_var * len_sequence_var * num_node_var, 4)
                arange = T.arange(num_node_var * size_batch_var * len_sequence_var * num_node_var)

                def step_make_indice(idx):
                    zeros = T.zeros(shape_indice)
                    floor, jnode = T.divmod(idx, num_node_var)
                    floor, iseq = T.divmod(floor, len_sequence_var)
                    inode, isample = T.divmod(floor, size_batch_var)
                    subtensor_zeros = zeros[idx]

                    return T.set_subtensor(subtensor_zeros, [inode, isample, iseq, jnode])

                results, _ = theano.scan(fn=step_make_indice, sequences=[arange], non_sequences=[])
                results = T.sum(results, axis=0)

                return results

            indice = make_indice(num_node, size_batch, len_sequence)

            def test_make_indice():
                x = numpy.random.rand(2, 1, 4, 3)
                h = numpy.ones((2, 1, 4, 5))

                fn_make_indice = theano.function(inputs=[samples], outputs=make_indice(num_node, size_batch, len_sequence),
                                                 allow_input_downcast=True)
                ans = fn_make_indice(x)

                assert ans.shape == (16, 4)

            # test_make_indice()

            """Constants"""

            # add extra 1 to each edge for clipping
            min_mn = T.constant(-(self.range_pool + 1), name='min_mn')
            max_mn = T.constant(self.range_pool + 1, name='max_mn')
            scale_pool = T.constant(self.scale_pool, name='scale_pool')

            def step_compute(idx, samples_arg, hids_arg):
                inode = T.basic._convert_to_int8(idx[0])
                isample = T.basic._convert_to_int8(idx[1])
                iseq = T.basic._convert_to_int8(idx[2])
                jnode = T.basic._convert_to_int8(idx[3])

                my_xy = samples_arg[inode, isample, iseq, -2:]
                your_xy = samples_arg[jnode, isample, iseq, -2:]
                your_hid = hids_arg[jnode, isample, iseq]  # (dim_hid,)
                your_hid = T._tensor_py_operators.dimshuffle(your_hid, ['x', 0])
                difference = your_xy - my_xy

                mn = T.int_div(difference, scale_pool)
                mn = T.basic._convert_to_uint8(mn)
                mn = T.clip(mn, min_mn, max_mn - 1)
                mn = T.add(mn, -min_mn)

                zeros = T.zeros((max_mn - min_mn, max_mn - min_mn, 1))
                indicator = T.set_subtensor(zeros[mn[0], mn[1], 0], 1)
                # cut the edges
                indicator = indicator[1:-1, 1:-1]

                ret = T.dot(indicator, your_hid)
                ret = T.cast(ret, dtype='float32')
                # (range_pool, range_pool, 1) * (dim_hid,) : (range_pool, range_pool, dim_hid)
                return ret

            # (num_node, size_batch, len_sequence, num_node, range_pool, range_pool, dim_hid))
            products, _ = theano.scan(fn=step_compute, sequences=[indice],
                                      non_sequences=[samples, relu_hids])

            shape_products = products.shape  # (*, range_pool, range_pool, dim_hid)
            products = T.reshape(products,
                                 newshape=(num_node, size_batch, len_sequence, num_node,
                                           shape_products[1], shape_products[2], shape_products[3]))
            social_tensors = T.sum(products, axis=3)

            def test_social_tensor():
                x = numpy.random.rand(2, 10, 4, 3)
                h = numpy.ones((2, 10, 4, 1))

                fn = theano.function(inputs=[samples, prev_hids], outputs=social_tensors, allow_input_downcast=True)
                ans = fn(x, h)

                assert ans.shape == (2, 1, 4, 10, 10, 5)

            # test_social_tensor()

            return social_tensors

        except:
            raise

    def reset_entry(self):
        self.entry_epoch = 0
        self.entry_batch = 0

    # todo change to pure theano
    def build_network(self, params=None):
        """
        Build computation graph from `inputs` to `outputs`, as well as `params` and so.
        :param params: Give certain parameter values (exported previously maybe) to build the network based on.
        :return: `outputs`, `params`
        """
        try:
            timer = utils.Timer()

            utils.xprint('Building Social LSTM network ... ')
            utils.xprint("using share scheme '%s' ... " % self.share_scheme)

            # IN = [(sec, x, y)]
            layer_in = L.layers.InputLayer(input_var=self.samples,
                                           shape=(self.num_node, None, self.length_sequence_input,
                                                  self.dimension_sample),
                                           name='input-layer')

            """
            Build embedding layer
            """

            # e = f_e(IN; W_e, b_e)
            layer_e = L.layers.DenseLayer(layer_in,
                                          num_units=self.dimension_embed_layer,
                                          W=self.w_e,
                                          b=self.b_e,
                                          nonlinearity=self.f_e,
                                          num_leading_axes=3,
                                          name='e-layer')
            assert utils.match(layer_e.output_shape,
                               (self.num_node, None, self.length_sequence_input, self.dimension_embed_layer))

            layer_embed = layer_e

            """
            Build LSTM hidden layers
            """

            # Prepare the input for hidden layers

            list_inputs_hidden = []

            # if self.share_scheme == 'social':
            #     # prev_hids: (num_node, batch_size, self.length_sequence_input, dim_hid)
            #     shape_prev_hid = (self.num_node, self.size_batch, self.length_sequence_input,
            #                       self.dimension_hidden_layer)
            #     layer_prev_hid = L.layers.InputLayer(input_var=None, name='prev-hid-layer', shape=shape_prev_hid)
            #     self.prev_hids = L.layers.get_output(layer_prev_hid)
            #     self.network_inputs += [self.prev_hids]
            #     social_tensor = self.compute_social_tensors(self.samples, self.prev_hids)
            #
            #     shape_social_tensor = (self.num_node, self.size_batch, self.length_sequence_input,
            #                            2 * self.range_pool, 2 * self.range_pool, self.dimension_hidden_layer)
            #     layer_social_tensor = L.layers.InputLayer(input_var=social_tensor,
            #                                               shape=shape_social_tensor,
            #                                               name='input-social_tensor')
            #
            #     # a = f_a(Social Tensor; W_a, b_a)
            #     layer_a = L.layers.DenseLayer(layer_social_tensor,
            #                                   num_units=self.dimension_embed_layer,
            #                                   W=self.w_a,
            #                                   b=self.b_a,
            #                                   nonlinearity=self.f_a,
            #                                   num_leading_axes=3,
            #                                   name='a-layer')
            #     assert utils.match(layer_a.output_shape,
            #                        (self.num_node, None, self.length_sequence_input, self.dimension_embed_layer))
            #
            #     layer_embed = L.layers.ConcatLayer([layer_e, layer_a], axis=-1, name='embed-e&a-layer')
            #     assert utils.match(layer_embed.output_shape,
            #                        (self.num_node, None, self.length_sequence_input, 2 * self.dimension_embed_layer))

            if self.share_scheme == 'olstm':

                shape_occupancy_map = (self.num_node, self.size_batch, self.length_sequence_input,
                                       2 * self.range_pool, 2 * self.range_pool, 1)
                # occupancy_map = self.compute_occupancy_map(self.samples)
                # layer_occupancy_map = L.layers.InputLayer(input_var=occupancy_map,
                #                                           shape=shape_occupancy_map,
                #                                           name='input-occupancy-map')

                layer_occupancy_map = L.layers.ExpressionLayer(layer_in, self.compute_occupancy_map,
                                                               output_shape=shape_occupancy_map,
                                                               name='input-occupancy-map')
                # o = f_o(Occupancy Map; W_o, b_o)
                layer_o = L.layers.DenseLayer(layer_occupancy_map,
                                              num_units=self.dimension_embed_layer,
                                              W=self.w_o,
                                              b=self.b_o,
                                              nonlinearity=self.f_o,
                                              num_leading_axes=3,
                                              name='o-layer')
                assert utils.match(layer_o.output_shape,
                                   (self.num_node, None, self.length_sequence_input, self.dimension_embed_layer))

                layer_embed = L.layers.ConcatLayer([layer_e, layer_o], axis=-1, name='embed-e&o-layer')
                assert utils.match(layer_embed.output_shape,
                                   (self.num_node, None, self.length_sequence_input, 2 * self.dimension_embed_layer))

            # Share reshaped embedded inputs of all the nodes
            # , when using share scheme 'input'
            if self.share_scheme == 'input':
                _reshaped_embed = L.layers.ReshapeLayer(layer_embed,
                                                        shape=([1], [2], -1),
                                                        name='reshaped-embed-layer')
                for inode in xrange(0, self.num_node):
                    list_inputs_hidden += [_reshaped_embed]

            # Slice its own embedded input for each node
            # , when using the other share schemes
            else:
                for inode in xrange(0, self.num_node):
                    list_inputs_hidden += [L.layers.SliceLayer(layer_embed,
                                                               name='sliced-embed-layer[%d]' % inode,
                                                               indices=inode,
                                                               axis=0)]

            assert all(utils.match(_each_input_hidden.output_shape,
                                   (None, self.length_sequence_input, None))
                       for _each_input_hidden in list_inputs_hidden)

            # number of hidden layers
            n_hid = self.num_hidden_layer
            # dimension of lstm for each node in each hidden layer
            dim_hid = self.dimension_hidden_layer

            # Build unique lstm layer object for only 1 node in each hidden layer
            # , the other nodes would copy from this node
            # , when using share scheme 'input'
            if self.share_scheme == 'parameter':
                n_unique_lstm = 1
                n_shared_lstm = self.num_node - 1
            # Build unique lstm layer objects for all nodes
            # , when using share scheme 'social'
            else:
                n_unique_lstm = self.num_node
                n_shared_lstm = 0

            # list of layer objects for all hidden layer
            # each one concated from layer objects for all nodes in single hidden layer
            list2d_layers_hidden = []

            for ihid in xrange(0, n_hid):
                # list of layer objects for all nodes, in single hidden layer
                list_this_layer = []

                # Create a LSTMLayer object for each unique node

                for inode in xrange(0, n_unique_lstm):

                    _layer_input = list_inputs_hidden[inode] if ihid == 0 else list2d_layers_hidden[-1][inode]

                    _lstm = L.layers.LSTMLayer(_layer_input,
                                               num_units=dim_hid,
                                               nonlinearity=self.f_lstm_hid,
                                               hid_init=self.init_lstm_hid,
                                               cell_init=self.init_lstm_cell,
                                               grad_clipping=self.grad_clip,
                                               only_return_final=False,
                                               ingate=L.layers.Gate(W_in=self.w_lstm_in,
                                                                    W_hid=self.w_lstm_hid,
                                                                    W_cell=self.w_lstm_hid,
                                                                    b=self.b_lstm),
                                               forgetgate=L.layers.Gate(W_in=self.w_lstm_in,
                                                                        W_hid=self.w_lstm_hid,
                                                                        W_cell=self.w_lstm_hid,
                                                                        b=self.b_lstm),
                                               cell=L.layers.Gate(W_cell=None,
                                                                  nonlinearity=self.f_lstm_cell),
                                               outgate=L.layers.Gate(W_in=self.w_lstm_in,
                                                                     W_hid=self.w_lstm_hid,
                                                                     W_cell=self.w_lstm_hid,
                                                                     b=self.b_lstm),
                                               name="LSTM[%d,%d]" % (ihid + 1, inode + 1))

                    assert utils.match(_lstm.output_shape, (None, self.length_sequence_input, dim_hid))
                    list_this_layer += [_lstm]

                # the last unique LSTMLayer object to share from
                lstm_shared = list_this_layer[-1]

                # Create a LSTMLayer object for each shared node

                for inode in xrange(n_unique_lstm, n_unique_lstm + n_shared_lstm):

                    _layer_input = list_inputs_hidden[inode] if ihid == 0 else list2d_layers_hidden[-1][inode]

                    _lstm = L.layers.LSTMLayer(_layer_input,
                                               num_units=dim_hid,
                                               nonlinearity=self.f_lstm_hid,
                                               hid_init=self.init_lstm_hid,
                                               cell_init=self.init_lstm_cell,
                                               grad_clipping=self.grad_clip,
                                               only_return_final=False,
                                               ingate=L.layers.Gate(W_in=lstm_shared.W_in_to_ingate,
                                                                    W_hid=lstm_shared.W_hid_to_ingate,
                                                                    W_cell=lstm_shared.W_cell_to_ingate,
                                                                    b=lstm_shared.b_ingate),
                                               outgate=L.layers.Gate(W_in=lstm_shared.W_in_to_outgate,
                                                                     W_hid=lstm_shared.W_hid_to_outgate,
                                                                     W_cell=lstm_shared.W_cell_to_outgate,
                                                                     b=lstm_shared.b_outgate),
                                               forgetgate=L.layers.Gate(
                                                   W_in=lstm_shared.W_in_to_forgetgate,
                                                   W_hid=lstm_shared.W_hid_to_forgetgate,
                                                   W_cell=lstm_shared.W_cell_to_forgetgate,
                                                   b=lstm_shared.b_forgetgate),
                                               cell=L.layers.Gate(W_in=lstm_shared.W_in_to_cell,
                                                                  W_hid=lstm_shared.W_hid_to_cell,
                                                                  W_cell=None,
                                                                  b=lstm_shared.b_cell,
                                                                  nonlinearity=self.f_lstm_cell),
                                               name="LSTM[%d,%d]" % (ihid + 1, inode + 1))

                    assert utils.match(_lstm.output_shape, (None, self.length_sequence_input, dim_hid))
                    list_this_layer += [_lstm]

                pass  # end of building of single hidden layer for all nodes
                list2d_layers_hidden += [list_this_layer]

            pass  # end of building of multiple hidden layers

            # Dimshuffle & concate lstms in the last hidden layer into single layer object

            _list_last_hidden = list2d_layers_hidden[-1]
            list_shuffled_lstms = []
            for inode in xrange(self.num_node):
                # add an extra dim of 1 for concatation
                list_shuffled_lstms += [
                    L.layers.DimshuffleLayer(_list_last_hidden[inode],
                                             pattern=('x', 0, 1, 2))]

            layer_last_hid = L.layers.ConcatLayer(list_shuffled_lstms,
                                                  axis=0)
            assert utils.match(layer_last_hid.output_shape,
                               (self.num_node, None, self.length_sequence_input, dim_hid))

            layer_to_decode = layer_last_hid

            """
            Build the decoding layer
            """

            layer_means = L.layers.DenseLayer(layer_to_decode,
                                              num_units=2,
                                              W=self.w_means,
                                              b=self.b_means,
                                              nonlinearity=self.f_means,
                                              num_leading_axes=3,
                                              name="mean-layer")

            if self.decode_scheme == 'binorm':
                layer_deviations = L.layers.DenseLayer(layer_to_decode,
                                                       num_units=2,
                                                       W=self.w_deviations,
                                                       b=self.b_deviations,
                                                       nonlinearity=self.f_deviations,
                                                       num_leading_axes=3,
                                                       name="deviation-layer")
                layer_correlation = L.layers.DenseLayer(layer_to_decode,
                                                        num_units=1,
                                                        W=self.w_correlation,
                                                        b=self.b_correlation,
                                                        nonlinearity=self.f_correlation,
                                                        num_leading_axes=3,
                                                        name="correlation-layer")
                layer_distribution = L.layers.ConcatLayer([layer_means, layer_deviations, layer_correlation],
                                                          axis=-1,
                                                          name="distribution-layer")

                assert utils.match(layer_distribution.output_shape,
                                   (self.num_node, None, self.length_sequence_input, 5))
                layer_decoded = layer_distribution

            elif self.decode_scheme == 'euclidean':
                assert utils.match(layer_means.output_shape,
                                   (self.num_node, None, self.length_sequence_input, 2))
                layer_decoded = layer_means
            else:
                raise ValueError("No definition found for loss scheme '%s'." % self.decode_scheme)

            """
            Build final output layer
            """

            # Slice x-length sequence from the last, according to <length_sequence_output>
            layer_sliced = L.layers.SliceLayer(layer_decoded,
                                               name='sliced-layer',
                                               indices=slice(-self.length_sequence_output, None),
                                               axis=2)
            layer_out = L.layers.ExpressionLayer(layer_sliced,
                                                 lambda x: x[:, :, ::-1, :],
                                                 output_shape=layer_sliced.output_shape,
                                                 name='output-layer')

            """
            Save some useful variables
            """

            self.embed = L.layers.get_output(layer_embed)
            self.hids_all = []
            for ilayer in xrange(len(list2d_layers_hidden)):
                self.hids_all += [[]]
                self.hids_all[ilayer] += L.layers.get_output(list2d_layers_hidden[ilayer])
            self.hid_last = L.layers.get_output(layer_last_hid)

            self.network = layer_out
            self.network_outputs = [L.layers.get_output(layer_out)]
            self.params_all = L.layers.get_all_params(layer_out)
            self.params_trainable = L.layers.get_all_params(layer_out, trainable=True)

            def get_names_for_params(_list_params):
                utils.assertor.assert_type(_list_params, list)
                param_keys = []
                for _param in _list_params:
                    # utils.assertor.assert_type(param, TensorSharedVariable)
                    _name = _param.name
                    param_keys += [_name]
                return param_keys

            self.param_names = get_names_for_params(self.params_all)

            """
            Import external paratemers if given
            """

            # Assign saved parameter values to the built network
            if params is not None:
                utils.xprint('importing given parameters ... ')
                self.set_params(params)
                picklename = 'params-imported.pkl'
            else:
                picklename = 'params-init.pkl'

            # Save initial values of params for possible future restoration
            self.initial_param_values = L.layers.get_all_param_values(layer_out)
            self.current_param_values = self.initial_param_values
            # export initial param values for record
            self.export_params(filename=picklename)

            utils.xprint('done in %s.' % timer.stop(), newline=True)
            return self.network_outputs, self.params_all

        except:
            raise

    def compute_and_compile(self):
        """
        Would force to redo from decoding to compiling.
        :return:
        """
        try:
            self.predictions = None
            self._compute_prediction_()

            self.loss = None
            self._compute_loss_()

            self.deviations = None
            self._compute_deviation_()

            self.updates = None
            self._compute_update_()

            self.func_predict = None
            self.func_compare = None
            self.func_train = None
            self.peek_outputs = None
            self.peek_params = None
            self._compile_function_()

        except:
            raise

    def _compute_prediction_(self):
        """
        Build computation graph from `outputs` to `predictions`, only if `predictions` is None.
        :return:
        """
        try:
            utils.assertor.assert_not_none(self.network_outputs, "Must build the network first.")

            utils.xprint('Decoding ... ')

            if self.predictions is None:
                outputs = self.network_outputs[0]
                # Use mean(x, y) as predictions directly
                predictions = outputs[:, :, :, 0:2]

                self.predictions = predictions
                utils.xprint('done.', newline=True)
            else:
                utils.xprint('skipped.', newline=True)

        except:
            raise

    def _compute_loss_(self):
        """
        (NNL) bivariant normal loss of Euclidean distance loss for training.
        Build computation graph from `predictions`, `targets` to `loss`, only if `loss` is None.
        :return:
        """
        try:
            utils.assertor.assert_not_none(self.network_outputs, "Must build the network first.")
            utils.assertor.assert_not_none(self.predictions, "Must compute the prediction first.")

            utils.xprint('Computing loss ... ')

            if self.loss is None:
                # Remove time column
                facts = self.targets[:, :, :, -2:]
                shape_facts = facts.shape
                shape_stacked_facts = (shape_facts[0] * shape_facts[1] * shape_facts[2], shape_facts[3])

                # Use either (nnl) binorm or euclidean distance for loss

                loss = None

                if self.decode_scheme == 'binorm':
                    """
                    NNL Bivariant normal distribution
                    """
                    utils.xprint("using decode scheme 'binorm' ... ")

                    # Reshape for convenience
                    facts = T.reshape(facts, shape_stacked_facts)

                    outputs = self.network_outputs[0]
                    shape_distributions = outputs.shape
                    shape_stacked_distributions = (shape_distributions[0] * shape_distributions[1] * shape_distributions[2],
                                                   shape_distributions[3])
                    distributions = T.reshape(outputs, shape_stacked_distributions)

                    # Use scan to replace loop with tensors
                    def step_loss(idx, distribution_mat, fact_mat):

                        # From the idx of the start of the slice, the vector and the length of
                        # the slice, obtain the desired slice.

                        distribution = distribution_mat[idx, :]
                        means = distribution[0:2]
                        stds = distribution[2:4]
                        correlation = distribution[5]
                        target = fact_mat[idx, :]
                        prob = SocialLSTM.bivar_norm(target[0], target[1], means[0], means[1], stds[0], stds[1],
                                                     correlation)

                        # Do something with the slice here.

                        return prob

                    def step_loss_scaled(idx, distribution_mat, fact_mat, motion_range_v):

                        # From the idx of the start of the slice, the vector and the length of
                        # the slice, obtain the desired slice.

                        distribution = distribution_mat[idx, :]
                        means = distribution[0:2]
                        scaled_stds = T.mul(distribution[2:4], motion_range_v)
                        deviations = T.mul(distribution[2:4], motion_range_v)
                        correlation = distribution[4]
                        target = fact_mat[idx, :]
                        prob = SocialLSTM.bivar_norm(target[0], target[1], means[0], means[1], scaled_stds[0], scaled_stds[1],
                                                     correlation)

                        # Do something with the slice here.

                        return prob

                    # Make a vector containing the start idx of every slice
                    indices = T.arange(facts.shape[0])

                    if self.scaled_sigma:
                        motion_range = T.constant(self.motion_range[1] - self.motion_range[0])
                        probs, updates_loss = theano.scan(fn=step_loss_scaled, sequences=[indices],
                                                          non_sequences=[distributions, facts, motion_range])
                    else:
                        probs, updates_loss = theano.scan(fn=step_loss, sequences=[indices],
                                                          non_sequences=[distributions, facts])

                    # save binorm probabilities for future peeking
                    self.probabilities = probs

                    # Normal Negative Log-likelihood
                    nnls = T.neg(T.log(probs))
                    loss = nnls

                elif self.decode_scheme == 'euclidean':
                    """
                    Euclidean distance loss
                    """
                    utils.xprint("using decode scheme 'euclidean' ... ")

                    # Elemwise differences
                    differences = T.sub(self.predictions, facts)
                    differences = T.reshape(differences, shape_stacked_facts)
                    deviations = T.add(differences[:, 0] ** 2, differences[:, 1] ** 2) ** 0.5
                    shape_deviations = (shape_facts[0], shape_facts[1], shape_facts[2])
                    deviations = T.reshape(deviations, shape_deviations)

                    loss = deviations

                else:
                    raise ValueError("No definition found for loss scheme '%s'." % self.decode_scheme)

                # Compute mean for loss
                loss = T.mean(loss)

                self.loss = loss
                utils.xprint('done.', newline=True)
            else:
                utils.xprint('skipped.', newline=True)

        except:
            raise

    def _compute_deviation_(self):
        """
        Euclidean Distance for Observation.
        Build computation graph from `predictions`, `targets` to `deviations`, only if `deviations` is None.
        :return:
        """
        try:
            utils.assertor.assert_not_none(self.network_outputs, "Must build the network first.")
            utils.assertor.assert_not_none(self.predictions, "Must compute the prediction first.")

            utils.xprint('Computing deviation for observation ... ')

            if self.deviations is None:
                # Remove time column
                facts = self.targets[:, :, :, -2:]
                shape_facts = facts.shape
                shape_stacked_facts = (shape_facts[0] * shape_facts[1] * shape_facts[2], shape_facts[3])

                # Elemwise differences
                differences = T.sub(self.predictions, facts)
                differences = T.reshape(differences, shape_stacked_facts)
                deviations = T.add(differences[:, 0] ** 2, differences[:, 1] ** 2) ** 0.5
                shape_deviations = (shape_facts[0], shape_facts[1], shape_facts[2])
                deviations = T.reshape(deviations, shape_deviations)

                self.deviations = deviations
                utils.xprint('done.', newline=True)
            else:
                utils.xprint('skipped.', newline=True)

        except:
            raise

    def _compute_update_(self):
        """
        RMSProp training.
        Build computation graph from `loss` to `updates`, only if `updates` is None.
        :return:
        """
        try:
            utils.assertor.assert_not_none(self.network_outputs, "Must build the network first.")
            utils.assertor.assert_not_none(self.predictions, "Must compute the prediction first.")
            utils.assertor.assert_not_none(self.deviations, "Must compute the deviation first.")

            utils.xprint('Computing updates ... ')

            if self.updates is None:
                utils.xprint("using train scheme '%s' ... " % self.train_scheme)

                # Compute updates according to given training scheme
                if self.train_scheme == 'rmsprop':
                    updates = L.updates.rmsprop(self.loss, self.params_trainable, learning_rate=self.learning_rate,
                                                rho=self.rho, epsilon=self.epsilon)
                elif self.train_scheme == 'adagrad':
                    updates = L.updates.adagrad(self.loss, self.params_trainable, learning_rate=self.learning_rate,
                                                epsilon=self.epsilon)
                elif self.train_scheme == 'momentum':
                    updates = L.updates.momentum(self.loss, self.params_trainable, learning_rate=self.learning_rate,
                                                 momentum=self.momentum)
                elif self.train_scheme == 'nesterov':
                    updates = L.updates.nesterov_momentum(self.loss, self.params_trainable,
                                                          learning_rate=self.learning_rate,
                                                          momentum=self.momentum)
                else:
                    raise ValueError("No definition found  for training scheme '%s'." % self.train_scheme)

                utils.assertor.assert_not_none(updates, "Computation of updates has failed.")

                self.updates = updates
                utils.xprint('done.', newline=True)
            else:
                utils.xprint('skipped ...')

        except:
            raise

    def _compile_function_(self):
        """
        Compile theano functions used for prediction, observation & training, only for those functions who are None.
        :return:
        """
        try:
            utils.assertor.assert_not_none(self.network_outputs, "Must build the network first.")
            utils.assertor.assert_not_none(self.predictions, "Must compute the prediction first.")
            utils.assertor.assert_not_none(self.loss, "Must compute the loss first.")
            utils.assertor.assert_not_none(self.deviations, "Must compute the deviation first.")
            utils.assertor.assert_not_none(self.updates, "Must compute the updates first.")

            timer = utils.Timer()
            utils.xprint('Compiling functions ... ')

            """
            Compile theano functions for prediction, observation & training
            """

            if self.func_predict is None:
                self.func_predict = theano.function(self.network_inputs, self.predictions, allow_input_downcast=True)
            if self.func_compare is None:
                self.func_compare = theano.function(self.network_inputs + [self.targets], self.deviations,
                                                    allow_input_downcast=True)
            if self.func_train is None:
                self.func_train = theano.function(self.network_inputs + [self.targets], self.loss, updates=self.updates,
                                                  allow_input_downcast=True)

            """
            Compile peeking functions for debugging
            """
            if self.peek_outputs is None:
                self.peek_outputs = theano.function(self.network_inputs, self.network_outputs, allow_input_downcast=True)
            if self.peek_params is None:
                self.peek_params = theano.function([], self.params_all, allow_input_downcast=True)

            utils.xprint('done in %s.' % timer.stop(), newline=True)

        except:
            raise

    def _train_single_batch_(self, batch, tag_log='train', with_target=True):
        """

        :param batch: (instants, samples, instants, targets)
        :param with_target: Whether to train with `targets`. Would train with predictions & use targets only for
                            comparison if `False`.
        :return:
        """

        instants_sample, samples, instants_target, targets = batch[0], batch[1], batch[2], batch[3]
        size_this_batch = len(instants_sample)

        # Disallowing interrupt once starting training
        # Keep trying until (1) done & return (2) exception caught

        loss_batch = None
        while True:
            try:
                # Predict before training
                predictions_batch = self.func_predict(samples)
                deviations_batch = self.func_compare(samples, targets) if targets is not None else None

                # use predictions as targets
                if with_target:
                    train_targets = targets
                else:
                    train_targets = predictions_batch

                # Actually do training, & only once

                while loss_batch is None:
                    try:
                        loss_batch = self.func_train(samples, train_targets)
                    except KeyboardInterrupt:
                        pass

                # Validate loss

                try:
                    utils.assertor.assert_finite(loss_batch, 'loss')

                except AssertionError, e:
                    raise utils.InvalidTrainError("Get loss of 'inf'.")

                # Validate params after training

                new_params = self.peek_params()
                try:
                    utils.assertor.assert_finite(new_params, 'params')

                except AssertionError, e:
                    raise utils.InvalidTrainError(
                        "Get parameters containing 'nan' or 'inf' after training.",
                        details="Parameters:\n"
                                "%s" % new_params)
                else:
                    self.current_param_values = new_params

                # Done & break
                break

            except utils.InvalidTrainError, e:
                raise
            except KeyboardInterrupt:
                pass
            except:
                raise

        # Logging

        _done_logging = False
        while not _done_logging:
            try:
                # Log [deviation, prediction, target] by each sample

                def log_by_sample():
                    # log as .trace format for convenience in analyse
                    FILENAME_COMPARE = '%s-epoch%d' % (tag_log, self.entry_epoch)
                    logname_compare = utils.filer.format_subpath(utils.get_config('path_compare'), FILENAME_COMPARE)
                    compare_content = ''

                    for isample in xrange(0, size_this_batch):

                        dict_content = {'epoch': self.entry_epoch, 'batch': self.entry_batch, 'sample': isample + 1}

                        for iseq in xrange(0, self.length_sequence_output):
                            # index in [-n, -1]
                            _instant = instants_target[
                                isample, iseq - self.length_sequence_output]
                            dict_content['instant'] = _instant
                            compare_content += "%d\t" % _instant

                            for inode in xrange(0, self.num_node):
                                # [x, y]
                                if inode > 0:
                                    compare_content += "\t"

                                _prediction = predictions_batch[inode, isample, iseq]
                                _prediction = "%.2f\t%.2f" % (_prediction[-2], _prediction[-1])
                                if targets is not None:
                                    _deviation = deviations_batch[inode, isample, iseq]
                                    _deviation = "%.2f" % _deviation
                                    _target = targets[inode, isample, iseq, -2:]
                                    _target = "%.2f\t%.2f" % (_target[-2], _target[-1])
                                    dict_content[self.node_identifiers[inode]] = "(%s\t[%s]\t[%s])" % (_deviation, _prediction, _target)
                                    compare_content += "%s\t%s" % (_prediction, _target)
                                else:
                                    dict_content[self.node_identifiers[inode]] = "[%s]" % _prediction
                                    compare_content += "%s" % _prediction

                            logname_sample = "%s-sample" % tag_log
                            self.logger.log(dict_content, name=logname_sample)
                            compare_content += "\n"

                    self.logger.log(compare_content, name=logname_compare)

                log_by_sample()

                if deviations_batch is not None:
                    # Print loss & deviation info to console
                    hitrates = self.compute_hitrate(deviations_batch, nbin=2, formatted=True)
                    utils.xprint('%s; %s; %s;'
                                 % (utils.format_var(float(loss_batch), name='loss'),
                                    utils.format_var(hitrates, name='hitrate'),
                                    utils.format_var(deviations_batch, name='deviations')),
                                 newline=True)

                # Log [loss, mean-deviation, min-deviation, max-deviation] by each batch

                def log_by_batch():
                    _peek_deviations_this_batch = utils.peek_matrix(deviations_batch, formatted=True) \
                        if deviations_batch is not None else None

                    self.logger.log({'epoch': self.entry_epoch, 'batch': self.entry_batch,
                                     'loss': utils.format_var(float(loss_batch)),
                                     'mean-deviation': _peek_deviations_this_batch['mean']
                                     if _peek_deviations_this_batch is not None else None,
                                     'min-deviation': _peek_deviations_this_batch['min']
                                     if _peek_deviations_this_batch is not None else None,
                                     'max-deviation': _peek_deviations_this_batch['max']
                                     if _peek_deviations_this_batch is not None else None},
                                    name="%s-batch" % tag_log)

                log_by_batch()

                # Done & break
                break

            except KeyboardInterrupt, e:
                pass
            except:
                raise
        pass  # end of while not _done_logging

        return predictions_batch, deviations_batch, loss_batch

    # todo add train_some

    def _train_single_epoch_(self, sampler, tag_log='train', with_target=True):

        FILENAME_COMPARE = '%s-epoch%d' % (tag_log, self.entry_epoch)
        logname_compare = utils.filer.format_subpath(utils.get_config('path_compare'), FILENAME_COMPARE)
        self.logger.register(logname_compare, overwritable=False)

        losses_epoch = numpy.zeros((0,))
        deviations_epoch = numpy.zeros((0,))
        hitrates_epoch = None

        done_batch = None  # Whether a batch has got finished properly
        # loop for each batch in single epoch
        while True:
            # start of single batch
            try:
                predictions, deviations, loss = None, None, None

                # retrieve the next batch for nodes
                # only if the previous batch is completed
                # else, redo the previous batch
                if done_batch is None \
                        or done_batch:
                    done_batch = False
                    instants_sample, samples, instants_target, targets = sampler.load_batch()

                # break if cannot find a new batch
                if targets is None:
                    if self.entry_batch == 0:
                        raise RuntimeError("Only %d sample pairs are found, "
                                           "not enough for one single batch of size %d."
                                           % (sampler.npair() - sampler.leng, self.size_batch))
                    break

                self.entry_batch += 1
                utils.xprint('    Batch %d ... ' % self.entry_batch)

                predictions, deviations, loss = self._train_single_batch_(
                    batch=(instants_sample, samples, instants_target, targets), tag_log=tag_log, with_target=with_target)

                # consider successful if training is done and successful
                done_batch = True

            except KeyboardInterrupt, e:

                _menu = [['stop', 's'],
                         ['continue', 'c'],
                         ['peek', 'p']]
                _hint = "0: (s)top & exit   1: (c)ontinue    2: (p)eek network output"

                utils.xprint('\n', newline=True)
                _choice = utils.ask(_hint, code_quit='q', interpretor=utils.interpret_menu, menu=_menu)
                utils.xprint('', newline=True)

                while _choice == 'peek':
                    _netout = self.peek_outputs(samples)
                    utils.xprint('Network Output:\n%s\n' % _netout, newline=True)

                    # ask again after peeking
                    utils.xprint('', newline=True)
                    _choice = utils.ask(_hint, code_quit='q', interpretor=utils.interpret_menu, menu=_menu)
                    utils.xprint('', newline=True)

                if _choice == 'stop':
                    # means n incomplete epochs
                    self.num_epoch = self.entry_epoch
                    utils.update_config('num_epoch', self.entry_epoch, 'runtime', silence=False)
                    self.stop = True
                    break
                else:
                    continue
            except utils.InvalidTrainError, e:
                sampler.reset_entry()
                self.entry_batch = 0
                raise
                
            except:
                raise

            finally:

                if done_batch:

                    losses_epoch = numpy.append(losses_epoch, loss)
                    deviations_epoch = numpy.append(deviations_epoch,
                                                    numpy.reshape(deviations, newshape=(-1, deviations.shape[-1])))
                    hitrates_epoch = self.compute_hitrate(deviations_epoch)

                else:  # skip logging if this batch is undone
                    pass

            pass  # end of single batch
        pass  # end of single epoch

        _done_logging = False
        while not _done_logging:
            try:
                _peek_losses_this_epoch = utils.peek_matrix(losses_epoch, formatted=True)
                _peek_deviations_this_epoch = utils.peek_matrix(deviations_epoch, formatted=True)

                # Print loss & deviation info to console
                utils.xprint('  mean-loss: %s; hitrate: %s; mean-deviation: %s'
                             % (_peek_losses_this_epoch['mean'],
                                self.compute_hitrate(deviations_epoch, nbin=2, formatted=True),
                                _peek_deviations_this_epoch['mean']),
                             newline=True)

                # Log [mean-loss, mean-deviation, min-deviation, max-deviation] by each epoch

                def log_by_epoch():
                    self.logger.log({'epoch': self.entry_epoch,
                                     'mean-loss': _peek_losses_this_epoch['mean'],
                                     'mean-deviation': _peek_deviations_this_epoch['mean'],
                                     'min-deviation': _peek_deviations_this_epoch['min'],
                                     'max-deviation': _peek_deviations_this_epoch['max']},
                                    name="%s-epoch" % tag_log)

                log_by_epoch()

                def log_hitrate():
                    self.logger.log({'epoch': self.entry_epoch,
                                     'hitrate': '%s' % hitrates_epoch},
                                    name="%s-hitrate" % tag_log)

                log_hitrate()
                _done_logging = True
            except KeyboardInterrupt:
                pass
            except:
                raise
            pass  # end of while not _done_logging

        sampler.reset_entry()
        self.entry_batch = 0
        return losses_epoch, deviations_epoch, hitrates_epoch

    def tryout(self, sampler):

        utils.assertor.assert_not_none(self.network_outputs, "Must build the network first.")
        utils.assertor.assert_not_none(self.predictions, "Must compute the prediction first.")
        utils.assertor.assert_not_none(self.loss, "Must compute the loss first.")
        utils.assertor.assert_not_none(self.deviations, "Must compute the deviation first.")
        for _func in (self.func_predict, self.func_compare):
            utils.assertor.assert_not_none(_func, "Must compile the functions first.")

        try:
            utils.xprint('Testing ... ', newline=True)
            timer = utils.Timer()

            # backup current param values
            params_original = self.current_param_values

            with_target = utils.get_config('tryout_with_target')
            _, deviations, hitrates = self._train_single_epoch_(sampler, tag_log='test', with_target=with_target)
            # must not change training entry
            # self.entry_epoch += 1

            # Save as the best params if necessary

            if hitrates is not None:
                if self.best_param_values['record'] is None \
                        or hitrates[0][1] >= self.best_param_values['record']:
                    self.update_best_params(self.entry_epoch, self.current_param_values, hitrates[0][1])

            # restore original param values after testing
            self.set_params(params_original)

            # Print deviation info to console
            # utils.xprint('  mean-deviation: %s' % numpy.mean(deviations_by_batch), newline=True)

            utils.xprint('Done in %s.' % timer.stop(), newline=True)
            self.export_params(overwritable=False)
            return deviations, hitrates

        except utils.InvalidTrainError, e:
            utils.xprint(e.message, newline=True)
            utils.xprint('Undone.', newline=True)
            
        except:
            raise

    def train(self, sampler, num_epoch=None):
        """

        :return: 2 <ndarray> containing average loss & deviation of each epoch.
        """
        utils.assertor.assert_not_none(self.network_outputs, "Must build the network first.")
        utils.assertor.assert_not_none(self.predictions, "Must compute the prediction first.")
        utils.assertor.assert_not_none(self.loss, "Must compute the loss first.")
        utils.assertor.assert_not_none(self.deviations, "Must compute the deviation first.")
        for _func in (self.func_predict, self.func_compare, self.func_train):
            utils.assertor.assert_not_none(_func, "Must compile the functions first.")

        # do the epochs all at once if not specified
        if num_epoch is None:
            num_epoch = self.num_epoch

        done_training = False
        while not done_training:
            # start of single training try
            try:
                utils.xprint('Training ... ', newline=True)
                timer = utils.Timer()

                losses_by_epoch = numpy.zeros((0,))
                deviations_by_epoch = numpy.zeros((0,))

                # for iepoch in range(num_epoch):
                iepoch = 0
                while True:
                    # start of single epoch
                    self.entry_epoch += 1
                    utils.xprint('  Epoch %d ... ' % self.entry_epoch, newline=True)

                    losses_by_batch, deviations_by_batch, hitrates_this_epoch = self._train_single_epoch_(sampler)

                    losses_by_epoch = numpy.append(losses_by_epoch, numpy.mean(losses_by_batch))
                    deviations_by_epoch = numpy.append(deviations_by_epoch, numpy.mean(deviations_by_batch))
                    iepoch += 1

                    if self.stop:
                        break
                    if iepoch >= num_epoch:
                        break

                pass  # end of all epochs
                done_training = True

            except utils.InvalidTrainError, e:

                utils.warn(e.message)

                # Update training parameters & Retrain if possible
                if self.adapt_train_parameter() is True:

                    sampler.reset_entry()
                    continue

                else:
                    raise

            pass  # end of single training attempt
        pass  # end of while not done_training

        self.logger.log_config()

        if not self.stop:
            utils.xprint('Done in %s.' % timer.stop(), newline=True)
        return losses_by_epoch, deviations_by_epoch

    def export_params(self, params=None, filename=None, replace=None, overwritable=True):
        """

        :param params: Parameter values to export. Current values are used if not specified.
        :param filename: Filename to export to. Default format is used if not specified.
        :return: Path of the exported file.
        """
        try:
            FILENAME_DEFAULT = 'params-epoch%d.pkl' % self.entry_epoch

            utils.assertor.assert_not_none(self.params_all, "Must build the network first.")

            if params is not None \
                    and filename is None:
                raise RuntimeError("A `filename` is requested with `params` given.")

            if params is None:
                params = self.current_param_values
            if filename is None:
                filename = FILENAME_DEFAULT

            path = self.logger.log_pickle(params, filename, replace=replace, overwritable=overwritable)
            return path

        except:
            raise

    def update_best_params(self, epoch, value, record):
        try:
            filename = 'params-best-epoch%d.pkl' % epoch

            old_path = self.best_param_values['path']
            path = self.export_params(params=value, filename=filename, replace=old_path, overwritable=False)

            self.best_param_values['epoch'] = epoch
            self.best_param_values['value'] = value
            self.best_param_values['record'] = record
            self.best_param_values['path'] = path

            utils.xprint("Best parameters have been exported to '%s'." % path, newline=True)
            return path

        except:
            raise

    def import_params(self, path=None):
        try:
            utils.assertor.assert_not_none(self.params_all, "Must build the network first.")

            if path is None:
                path = utils.ask('Import from file path?', interpretor=utils.interpret_file_path)
                if path is None:
                    return
            utils.xprint('Importing given parameters ... ')
            params_all = utils.filer.load_from_file(path)
            self.reset_params(params_all)

            utils.xprint('done.', newline=True)

        except:
            raise

    def set_params(self, params):
        try:
            L.layers.set_all_param_values(self.network, params)
            self.current_param_values = params

        except:
            raise

    def reset_params(self, params):
        try:
            L.layers.set_all_param_values(self.network, params)

            # Save initial values of params for possible future restoration
            self.initial_param_values = params
            self.current_param_values = self.initial_param_values
            self.best_param_values['epoch'] = None
            self.best_param_values['value'] = None
            self.best_param_values['record'] = None
            self.best_param_values['path'] = None

        except:
            raise

    def adapt_train_parameter(self):
        """
        Adapt gradient clipping or learning rate if possible.
        :return: True or False
        """
        try:
            # Update gradient clipping as 1st choice
            # , unless grad_clip <= 100
            if self.adaptive_grad_clip is not None \
                    and self.grad_clip > 100:

                # decay the decrement if necessary
                if -self.adaptive_grad_clip >= self.grad_clip:
                    self.adaptive_grad_clip /= 10
                    utils.update_config('adaptive_grad_clip', self.adaptive_grad_clip, 'runtime',
                                        silence=False)
                new_grad_clip = self.grad_clip + self.adaptive_grad_clip

                self._update_grad_clip_(new_grad_clip)

                # Reinitialize related variables
                self.reset_entry()

                return True

            # Update learning rate
            elif self.adaptive_learning_rate is not None:

                # update by decay ratio
                if self.adaptive_learning_rate > 0:
                    new_learning_rate = self.learning_rate * self.adaptive_learning_rate
                # update by decrement
                else:
                    # decay the decrement if necessary
                    if -self.adaptive_learning_rate >= self.learning_rate:
                        self.adaptive_learning_rate /= 10
                        utils.update_config('adaptive_learning_rate', self.adaptive_learning_rate, 'runtime',
                                            silence=False)
                    new_learning_rate = self.learning_rate + self.adaptive_learning_rate

                self._update_learning_rate_(new_learning_rate)

                # Reinitialize related variables
                self.reset_entry()

                return True

            else:
                return False

        except:
            raise

    def _update_learning_rate_(self, new_learning_rate):
        """
        1) Redo build-compute-compile if necessary. 2) Update log & config. 3) Restore initial parameters.
        :param new_learning_rate:
        :return:
        """
        try:
            self.learning_rate = new_learning_rate
            utils.update_config('learning_rate', new_learning_rate, source='runtime', silence=False)

            utils.xprint("Re")
            self.updates = None
            self._compute_update_()

            utils.xprint("Re")
            self.func_train = None
            self._compile_function_()

            utils.assertor.assert_not_none(self.initial_param_values,
                                           "No initial values are found for parameter restoration.")

            utils.xprint("Restore parameters from initial values ... ")
            self.reset_params(self.initial_param_values)
            utils.xprint("done.", newline=True)

        except:
            raise

    def _update_grad_clip_(self, new_grad_clip):
        """
        1) Redo build-compute-compile if necessary. 2) Update log & config. 3) Restore initial parameters.
        :param new_grad_clip:
        :return:
        """
        try:
            self.grad_clip = new_grad_clip
            utils.update_config('grad_clip', new_grad_clip, source='runtime', silence=False)

            utils.xprint("Re")
            self.build_network()

            self.compute_and_compile()

            utils.assertor.assert_not_none(self.initial_param_values,
                                           "No initial values are found for parameter restoration.")

            utils.xprint("Restore parameters from initial values ... ")
            self.reset_params(self.initial_param_values)
            utils.xprint("done.", newline=True)

        except:
            raise

    def complete(self):
        self.logger.complete()

    @staticmethod
    def test():

        try:

            # Select certain nodes if requested
            nodes = utils.get_config('nodes') if utils.has_config('nodes') else None
            nodes = utils.get_config('num_node') if nodes is None and utils.has_config('num_node') else nodes

            # Build sampler
            sampler = Sampler(nodes=nodes, keep_positive=True)
            sample_gridding = utils.get_config('sample_gridding')
            if sample_gridding is True:
                sampler.map_to_grid(grid_system=GridSystem(utils.get_config('scale_grid')))
            # Devide into train set & test set
            trainset = utils.get_config('trainset')
            sampler_trainset, sampler_testset = sampler.devide(trainset)
            # sampler_testset.with_target = False
            utils.xprint("Use %d samples as train set & %d samples as test set."
                         % (sampler_trainset.length(), sampler_testset.length()), newline=True)

            # Define the model
            model = SocialLSTM(node_identifiers=sampler.node_identifiers, motion_range=sampler.motion_range)
            ask = utils.get_config('ask')

            try:
                # Import previously pickled parameters if requested
                file_unpickle = utils.get_config('file_unpickle') if utils.has_config('file_unpickle') else None
                params_unpickled = utils.filer.load_from_file(file_unpickle) if file_unpickle is not None else None

                # Build & compile the model
                model.build_network(params=params_unpickled)
                model.compute_and_compile()

                if params_unpickled is not None:
                    model.tryout(sampler_testset)

                train_losses = numpy.zeros((0,))
                train_deviations = numpy.zeros((0,))

                if model.num_epoch > 0:
                    # Do training
                    try:

                        while True:
                            _losses, _deviations = model.train(sampler_trainset, num_epoch=utils.get_config('tryout_frequency'))
                            train_losses = numpy.append(train_losses, _losses)
                            train_deviations = numpy.append(train_deviations, _deviations)
                            if model.stop:
                                break

                            model.tryout(sampler_testset)

                            if model.stop:
                                break
                            if model.entry_epoch >= model.num_epoch:
                                if ask:

                                    more = utils.ask("Try more epochs?", code_quit=None, interpretor=utils.interpret_confirm)

                                    if more:

                                        num_more = utils.ask("How many?", interpretor=utils.interpret_positive_int)

                                        # quit means no more epochs
                                        if num_more is not None \
                                                and num_more > 0:
                                            model.num_epoch += num_more
                                            utils.update_config('num_epoch', model.num_epoch, 'runtime', silence=False)
                                        else:
                                            break
                                    # stop if no more
                                    else:
                                        break
                                # stop if not ask
                                else:
                                    break
                            else:
                                continue

                    except Exception, e:
                        utils.handle(e)
                    else:
                        pass

                    finally:
                        pass

                else:  # tryout only
                    pass

            except:
                raise
            finally:
                model.complete()

        except Exception, e:
            raise
