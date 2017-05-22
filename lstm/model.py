# coding:utf-8

import numpy

import theano
import theano.tensor as T
from theano.tensor.sharedvar import TensorSharedVariable
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
                     'input']

    TRAIN_SCHEMES = ['rmsprop',
                     'adagrad',
                     'momentum',
                     'nesterov']

    DECODE_SCHEMES = ['binorm',
                      'euclidean']

    def __init__(self, node_identifiers, motion_range, inputs=None, targets=None, adaptive_learning_rate=None,
                 share_scheme=None, decode_scheme=None, train_scheme=None,
                 dimension_sample=None, length_sequence_input=None, length_sequence_output=None, size_batch=None,
                 dimension_embed_layer=None, dimension_hidden_layer=None,
                 learning_rate=None, rho=None, epsilon=None, momentum=None, grad_clip=None, num_epoch=None,
                 limit_network_history=None):
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
            self.dimension_hidden_layers = dimension_hidden_layer
            self.grad_clip = grad_clip if grad_clip is not None else utils.get_config('grad_clip')
            self.decode_scheme = decode_scheme if decode_scheme is not None else utils.get_config('decode_scheme')

            if self.decode_scheme not in SocialLSTM.DECODE_SCHEMES:
                raise ValueError("Unknown loss scheme '%s'. Must choose among %s." % (self.decode_scheme, SocialLSTM.DECODE_SCHEMES))

            self.train_scheme = train_scheme if train_scheme is not None else utils.get_config('train_scheme')
            if self.train_scheme not in SocialLSTM.TRAIN_SCHEMES:
                raise ValueError(
                    "Unknown training scheme '%s'. Must be among %s." % (self.train_scheme, SocialLSTM.TRAIN_SCHEMES))

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

            # Theano tensor variables (symbolic)

            if inputs is None:
                inputs = T.tensor4("input_var", dtype='float32')
            if targets is None:
                targets = T.tensor4("target_var", dtype='float32')

            # tensors, should take single batch of samples as real value
            self.inputs = inputs
            self.targets = targets

            self.embed = None  # tensor, output of embedding layer
            self.hids = []  # 2d list of tensors, output of all the hidden layers of all the nodes
            self.outputs = None  # tensor, output of the entire network
            self.network = None  # Lasagne layer object, containing the network structure
            self.params_all = None  # dict of tensors, all the parameters
            self.params_trainable = None  # dict of tensors, trainable parameters
            self.updates = None  # dict of tensors, returned from predefined training functions in Lasagne.updates

            # Real values stored for printing / debugging

            self.count_batch = 0  # real time batch counter for training
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
                'record': None,  # best record (mean deviation value) stored for comparison
                'value': None,  # actual param values
                'path': None}  # path of pickled file

            self.limit_network_history = limit_network_history if limit_network_history is not None \
                else utils.get_config('limit_network_history')
            # 2d list (iepoch, ibatch) of dict, data flow though the network
            if self.limit_network_history == 0:
                self.network_history = None
            else:
                self.network_history = []

            # Theano function objects

            self.func_predict = None
            self.func_compare = None
            self.func_train = None

            self.peek_embed = None
            self.peeks_hid = []
            self.peek_outputs = None
            self.peek_params = None
            self.peek_probs = None

            """
            Initialization Definitions
            """

            self.w_e = L.init.Uniform(std=0.005, mean=(1. / self.dimension_sample))
            self.b_e = L.init.Constant(0.)
            self.f_e = None

            self.w_lstm_in = L.init.Uniform(std=0.005, mean=(1. / self.dimension_embed_layer))
            self.w_lstm_hid = L.init.Uniform(std=0.005, mean=(1. / self.dimension_hidden_layers[1]))
            self.w_lstm_cell = L.init.Uniform(std=0.005, mean=(1. / self.dimension_hidden_layers[1]))
            self.b_lstm = L.init.Constant(0.)
            self.f_lstm_hid = L.nonlinearities.rectify
            self.f_lstm_cell = L.nonlinearities.rectify
            self.init_lstm_hid = L.init.Constant(0.)
            self.init_lstm_cell = L.init.Constant(0.)

            self.w_means = L.init.Uniform(std=0.005, mean=(1. / self.dimension_hidden_layers[1]))
            self.b_means = L.init.Constant(0.)
            self.f_means = None

            self.scaled_sigma = True

            self.w_deviations = L.init.Uniform(std=0.1, mean=(100. / self.dimension_hidden_layers[1] / self.num_node))
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
            self.logger.register("test-sample", columns=_columns)
            self.logger.register("test-batch", columns=['epoch', 'batch', 'loss',
                                                        'mean-deviation', 'min-deviation', 'max-deviation'])

            self.logger.register("test-epoch", columns=["epoch", "mean-loss",
                                                        "mean-deviation", "min-deviation", "max-deviation"])

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
            beta = T.as_tensor_variable(beta)
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

    def reset_entry(self):

        self.count_batch = 0  # real time batch counter for training
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

            # IN = [(sec, x, y)]
            layer_in = L.layers.InputLayer(input_var=self.inputs,
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
                                          name='embed-layer')
            assert utils.match(layer_e.output_shape,
                               (self.num_node, None, self.length_sequence_input, self.dimension_embed_layer))

            """
            Build LSTM hidden layers
            """

            # Prepare the input for hidden layers

            list_inputs_hidden = []

            # Share reshaped embedded inputs of all the nodes
            if self.share_scheme == 'input':
                _reshaped_embed = L.layers.ReshapeLayer(layer_e,
                                                        shape=([1], [2], -1),
                                                        name='reshaped-embed-layer')
                for inode in xrange(0, self.num_node):
                    list_inputs_hidden += [_reshaped_embed]

            # Slice its own embedded input for each node
            else:
                for inode in xrange(0, self.num_node):
                    list_inputs_hidden += [L.layers.SliceLayer(layer_e,
                                                               name='sliced-embed-layer[%d]' % inode,
                                                               indices=inode,
                                                               axis=0)]

            assert all(utils.match(_each_input_hidden.output_shape,
                                   (None, self.length_sequence_input, None))
                       for _each_input_hidden in list_inputs_hidden)

            # number of hidden layers
            n_hid = self.dimension_hidden_layers[0]
            # dimension of lstm for each node in each hidden layer
            dim_hid = self.dimension_hidden_layers[1]

            if self.share_scheme == 'parameter':
                n_unique_lstm = 1
                n_shared_lstm = self.num_node - 1
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

            """
            Build the decoding layer
            """
            layer_means = L.layers.DenseLayer(layer_last_hid,
                                              num_units=2,
                                              W=self.w_means,
                                              b=self.b_means,
                                              nonlinearity=self.f_means,
                                              num_leading_axes=3,
                                              name="mean-layer")

            if self.decode_scheme == 'binorm':
                layer_deviations = L.layers.DenseLayer(layer_last_hid,
                                                       num_units=2,
                                                       W=self.w_deviations,
                                                       b=self.b_deviations,
                                                       nonlinearity=self.f_deviations,
                                                       num_leading_axes=3,
                                                       name="deviation-layer")
                layer_correlation = L.layers.DenseLayer(layer_last_hid,
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
            layer_out = L.layers.SliceLayer(layer_decoded,
                                            name='output-layer',
                                            indices=slice(-self.length_sequence_output, None),
                                            axis=2)

            """
            Save some useful variables
            """

            self.embed = L.layers.get_output(layer_e)
            for ilayer in xrange(len(list2d_layers_hidden)):
                self.hids += [[]]
                for _lstm in list2d_layers_hidden[ilayer]:
                    self.hids[ilayer] += [L.layers.get_output(_lstm)]

            self.network = layer_out
            self.outputs = L.layers.get_output(layer_out)
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
                utils.xprint('Importing given parameters ... ')
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
            return self.outputs, self.params_all

        except:
            raise

    def compute_prediction(self):
        """
        Build computation graph from `outputs` to `predictions`.
        :return: `predictions`
        """
        try:
            utils.assertor.assert_not_none(self.outputs, "Must build the network first.")

            timer = utils.Timer()
            utils.xprint('Decoding ... ')

            # Use mean(x, y) as predictions directly
            predictions = self.outputs[:, :, :, 0:2]

            self.predictions = predictions
            utils.xprint('done in %s.' % timer.stop(), newline=True)
            return self.predictions

        except:
            raise

    def compute_loss(self):
        """
        (NNL) bivariant normal loss of Euclidean distance loss for training.
        Build computation graph from `predictions`, `targets` to `loss`.
        :return: `loss`
        """
        try:
            utils.assertor.assert_not_none(self.outputs, "Must build the network first.")
            utils.assertor.assert_not_none(self.predictions, "Must compute the prediction first.")

            timer = utils.Timer()
            utils.xprint('Computing loss ... ')

            # Remove time column
            facts = self.targets[:, :, :, 1:3]
            shape_facts = facts.shape
            shape_stacked_facts = (shape_facts[0] * shape_facts[1] * shape_facts[2], shape_facts[3])

            # Use either (nnl) binorm or euclidean distance for loss

            loss = None

            if self.decode_scheme == 'binorm':
                """
                NNL Bivariant normal distribution
                """
                # Reshape for convenience
                facts = T.reshape(facts, shape_stacked_facts)

                shape_distributions = self.outputs.shape
                shape_stacked_distributions = (shape_distributions[0] * shape_distributions[1] * shape_distributions[2],
                                               shape_distributions[3])
                distributions = T.reshape(self.outputs, shape_stacked_distributions)

                # Use scan to replace loop with tensors
                def step_loss(idx, distribution_mat, fact_mat):

                    # From the idx of the start of the slice, the vector and the length of
                    # the slice, obtain the desired slice.

                    distribution = distribution_mat[idx, :]
                    means = distribution[0:2]
                    # deviations = distribution[2:4]
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
                    # deviations = distribution[2:4]
                    deviations = T.mul(distribution[2:4], motion_range_v)
                    correlation = distribution[4]
                    target = fact_mat[idx, :]
                    prob = SocialLSTM.bivar_norm(target[0], target[1], means[0], means[1], deviations[0], deviations[1],
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
            utils.xprint('done in %s.' % timer.stop(), newline=True)
            return self.loss

        except:
            raise

    def compute_deviation(self):
        """
        Euclidean Distance for Observation.
        Build computation graph from `predictions`, `targets` to `deviations`.
        :return: `deviations`
        """
        try:
            utils.assertor.assert_not_none(self.outputs, "Must build the network first.")
            utils.assertor.assert_not_none(self.predictions, "Must compute the prediction first.")

            timer = utils.Timer()
            utils.xprint('Computing deviation for observation ... ')

            # Remove time column
            facts = self.targets[:, :, :, 1:3]
            shape_facts = facts.shape
            shape_stacked_facts = (shape_facts[0] * shape_facts[1] * shape_facts[2], shape_facts[3])

            # Elemwise differences
            differences = T.sub(self.predictions, facts)
            differences = T.reshape(differences, shape_stacked_facts)
            deviations = T.add(differences[:, 0] ** 2, differences[:, 1] ** 2) ** 0.5
            shape_deviations = (shape_facts[0], shape_facts[1], shape_facts[2])
            deviations = T.reshape(deviations, shape_deviations)

            self.deviations = deviations
            utils.xprint('done in %s.' % timer.stop(), newline=True)
            return self.deviations
        except:
            raise

    def compute_update(self):
        """
        RMSProp training.
        Build computation graph from `loss` to `updates`.
        :return: `updates`
        """
        try:
            utils.assertor.assert_not_none(self.outputs, "Must build the network first.")
            utils.assertor.assert_not_none(self.predictions, "Must compute the prediction first.")
            utils.assertor.assert_not_none(self.deviations, "Must compute the deviation first.")

            timer = utils.Timer()
            utils.xprint('Computing updates ... ')

            # Compute updates according to given training scheme
            updates = None
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
            utils.xprint('done in %s.' % timer.stop(), newline=True)
            return self.updates
        except:
            raise

    def compile(self):
        """
        Compile theano functions used for prediction, observation & training.
        :return: `func_predict`, `func_compare`, `func_train`
        """
        try:
            utils.assertor.assert_not_none(self.outputs, "Must build the network first.")
            utils.assertor.assert_not_none(self.predictions, "Must compute the prediction first.")
            utils.assertor.assert_not_none(self.loss, "Must compute the loss first.")
            utils.assertor.assert_not_none(self.deviations, "Must compute the deviation first.")
            utils.assertor.assert_not_none(self.updates, "Must compute the updates first.")

            timer = utils.Timer()
            utils.xprint('Compiling functions ... ')

            """
            Compile theano functions for prediction, observation & training
            """

            self.func_predict = theano.function([self.inputs], self.predictions, allow_input_downcast=True)
            self.func_compare = theano.function([self.inputs, self.targets], self.deviations, allow_input_downcast=True)
            self.func_train = theano.function([self.inputs, self.targets], self.loss, updates=self.updates,
                                              allow_input_downcast=True)
            # self.func_train = theano.function([self.inputs, self.targets], self.loss, updates=updates,
            #                                   allow_input_downcast=True,
            #                                   mode=NanGuardMode(nan_is_error=True,
            #                                                     inf_is_error=True,
            #                                                     big_is_error=True))

            """
            Compile peeking functions for debugging
            """

            self.peek_embed = theano.function([self.inputs], self.embed, allow_input_downcast=True)
            self.peeks_hid = []
            for ihid in self.hids:
                self.peeks_hid += [theano.function([self.inputs], ihid, allow_input_downcast=True)]
            self.peek_outputs = theano.function([self.inputs], self.outputs, allow_input_downcast=True)
            self.peek_params = theano.function([], self.params_all, allow_input_downcast=True)
            self.peek_probs = theano.function([self.inputs, self.targets], self.probabilities,
                                              allow_input_downcast=True) if self.probabilities is not None else None

            utils.xprint('done in %s.' % timer.stop(), newline=True)
            return self.func_predict, self.func_compare, self.func_train

        except:
            raise

    def get_peeks(self):

        return self.peek_embed, self.peeks_hid, self.peek_outputs, self.peek_params, self.peek_probs

    def _predict_single_batch_(self, inputs, targets=None):
        try:
            predictions = self.func_predict(inputs)
            deviations = self.func_compare(inputs, targets) if targets is not None else None
            return predictions, deviations
        except:
            raise

    def _train_single_batch_(self, batch, tag_log='train'):

        instants_input, inputs, instants_target, targets = batch[0], batch[1], batch[2], batch[3]

        # Prepare for training, allowing interrupt
        # Record params, flow of data & probabilities to network history BEFORE training

        try:
            def peek_netflow():
                _embed = self.peek_embed(inputs)
                _hids = []
                for _ihid in xrange(len(self.peeks_hid)):
                    _peek_hid = self.peeks_hid[_ihid]
                    _hids += [_peek_hid(inputs)]
                _netout = self.peek_outputs(inputs)

                dict_netflow = {'embed': _embed, 'hiddens': _hids, 'netout': _netout}
                return dict_netflow

            if self.network_history is not None:

                _netflow = peek_netflow()
                _probs = self.peek_probs(inputs, targets) if self.peek_probs is not None else None
                # note that record [i, j] contains variable values BEFORE this training
                self.network_history[-1].append({'params': self.current_param_values,
                                                 'netflow': _netflow,
                                                 'probs': _probs})

                # throw away overdue history
                if self.limit_network_history is not None \
                        and len(self.network_history) > self.limit_network_history:
                    self.network_history = \
                        self.network_history[-self.limit_network_history:]

            predictions, deviations = self._predict_single_batch_(inputs, targets)

        except:
            raise

        # Disallowing interrupt once starting training
        # Keep trying until (1) done & return (2) exception caught

        loss = None
        while True:
            try:
                # Actually do training, & only once

                while loss is None:
                    try:
                        loss = self.func_train(inputs, targets)
                    except KeyboardInterrupt:
                        pass

                # Validate loss

                try:
                    utils.assertor.assert_finite(loss, 'loss')

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
                    self.logger.register(logname_compare)
                    compare_content = ''

                    size_this_batch = len(instants_input)
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

                                _deviation = deviations[inode, isample, iseq]
                                _prediction = predictions[inode, isample, iseq]
                                _target = targets[inode, isample, iseq, -2:]
                                dict_content[self.node_identifiers[inode]] = "(%.2f\t[%.2f\t%.2f]\t[%.2f\t%.2f])" \
                                                                             % (_deviation, _prediction[0], _prediction[1],
                                                                                _target[0], _target[1])
                                compare_content += "%.2f\t%.2f\t%.2f\t%.2f" \
                                                   % (_prediction[0], _prediction[1], _target[0], _target[1])

                            logname_sample = "%s-sample" % tag_log
                            self.logger.log(dict_content, name=logname_sample)
                            compare_content += "\n"

                    self.logger.log(compare_content, name=logname_compare)

                log_by_sample()

                # Print loss & deviation info to console
                utils.xprint('%s; %s'
                             % (utils.format_var(float(loss), name='loss'),
                                utils.format_var(deviations, name='deviations')),
                             newline=True)

                # Log [loss, mean-deviation, min-deviation, max-deviation] by each batch

                def log_by_batch():
                    _peek_deviations_this_batch = utils.peek_matrix(deviations, formatted=True)

                    self.logger.log({'epoch': self.entry_epoch, 'batch': self.entry_batch,
                                     'loss': utils.format_var(float(loss)),
                                     'mean-deviation': _peek_deviations_this_batch['mean'],
                                     'min-deviation': _peek_deviations_this_batch['min'],
                                     'max-deviation': _peek_deviations_this_batch['max']},
                                    name="%s-batch" % tag_log)

                log_by_batch()

                # Done & break
                break

            except KeyboardInterrupt, e:
                pass
            except:
                raise
        pass  # end of while not _done_logging

        self.count_batch += 1
        return predictions, deviations, loss

    def _train_single_epoch_(self, sampler, tag_log='train'):

        # start of single epoch
        if self.network_history is not None:
            self.network_history.append([])

        losses_by_batch = numpy.zeros((0,))
        deviations_by_batch = numpy.zeros((0,))

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
                    instants_input, inputs, instants_target, targets = sampler.load_batch(with_target=True)

                # break if cannot find a new batch
                if inputs is None:
                    if self.entry_batch == 0:
                        raise RuntimeError("Only %d sample pairs are found, "
                                           "not enough for one single batch of size %d."
                                           % (sampler.length, self.size_batch))
                    break

                self.entry_batch += 1
                utils.xprint('    Batch %d ... ' % self.entry_batch)

                predictions, deviations, loss = self._train_single_batch_(
                    batch=(instants_input, inputs, instants_target, targets), tag_log=tag_log)

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
                    _netout = self.peek_outputs(inputs)
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
            except:
                raise

            finally:

                if done_batch:

                    losses_by_batch = numpy.append(losses_by_batch, loss)
                    deviations_by_batch = numpy.append(deviations_by_batch, numpy.mean(deviations))

                else:  # skip logging if this batch is undone
                    pass

            pass  # end of single batch
        pass  # end of single epoch

        _done_logging = False
        while not _done_logging:
            try:
                _peek_losses_this_epoch = utils.peek_matrix(losses_by_batch, formatted=True)
                _peek_deviations_this_epoch = utils.peek_matrix(deviations_by_batch, formatted=True)

                # Print loss & deviation info to console
                utils.xprint('  mean-loss: %s; mean-deviation: %s'
                             % (_peek_losses_this_epoch['mean'], _peek_deviations_this_epoch['mean']),
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
                _done_logging = True
            except KeyboardInterrupt:
                pass
            except:
                raise
            pass  # end of while not _done_logging

        self.entry_batch = 0
        return losses_by_batch, deviations_by_batch

    def tryout(self, sampler):

        utils.assertor.assert_not_none(self.outputs, "Must build the network first.")
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

            _, deviations_by_batch = self._train_single_epoch_(sampler, tag_log='test')
            sampler.reset_entry()
            # must not change training entry
            # self.entry_epoch += 1

            # restore original param values after testing
            self.set_params(params_original)

            # Print deviation info to console
            # utils.xprint('  mean-deviation: %s' % numpy.mean(deviations_by_batch), newline=True)

            utils.xprint('Done in %s.' % timer.stop(), newline=True)
            return deviations_by_batch

        except:
            raise

    def train(self, sampler, num_epoch=None):
        """

        :return: 2 <ndarray> containing average loss & deviation of each epoch.
        """
        utils.assertor.assert_not_none(self.outputs, "Must build the network first.")
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

                    losses_by_batch, deviations_by_batch = self._train_single_epoch_(sampler)

                    losses_by_epoch = numpy.append(losses_by_epoch, numpy.mean(losses_by_batch))
                    deviations_by_epoch = numpy.append(deviations_by_epoch, numpy.mean(deviations_by_batch))
                    sampler.reset_entry()
                    iepoch += 1

                    # Save as the best params if necessary

                    if self.best_param_values['record'] is None \
                            or numpy.mean(deviations_by_batch) <= self.best_param_values['record']:
                        self.update_best_params(self.entry_epoch, self.current_param_values, numpy.mean(deviations_by_batch))

                    if iepoch >= num_epoch:
                        break

                pass  # end of all epochs
                done_training = True

            except utils.InvalidTrainError, e:
                # Update learning rate & Retrain

                if self.adaptive_learning_rate is not None:
                    utils.warn(e.message)
                    new_learning_rate = self.learning_rate * self.adaptive_learning_rate
                    self.update_learning_rate(new_learning_rate)

                    # Reinitialize related variables
                    sampler.reset_entry()
                    self.reset_entry()
                    if self.network_history is not None:
                        self.network_history = []

                    continue

                else:
                    raise

            pass  # end of single training attempt
        pass  # end of while not done_training

        utils.xprint('Done in %s.' % timer.stop(), newline=True)
        self.export_params()
        return losses_by_epoch, deviations_by_epoch

    def export_history(self, path=None):
        try:
            if self.network_history is None \
                    or len(self.network_history) == 0:
                return

            PICKLE_NAME = 'history.pkl'

            if path is None:
                path = PICKLE_NAME

            path = self.logger.log_pickle(self.network_history, PICKLE_NAME)

            utils.xprint("\nRecent network history has been exported to '%s'." % path, newline=True)

            return path

        except:
            raise

    def export_params(self, params=None, filename=None):
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

            path = self.logger.log_pickle(params, filename)
            return path

        except:
            raise

    def update_best_params(self, epoch, value, record):
        try:
            self.best_param_values['epoch'] = epoch
            self.best_param_values['value'] = value
            self.best_param_values['record'] = record

            filename = 'params-best-epoch%d.pkl' % epoch

            path = self.export_params(params=value, filename=filename)

            old_path = self.best_param_values['path']
            if old_path is not None \
                    and old_path != path:
                utils.filer.remove_file(old_path)
            self.best_param_values['path'] = path
            utils.update_config('file_pickle', path, 'runtime', tags=['path'])

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
            self.best_param_values = {'epoch': 0, 'value': self.initial_param_values}

        except:
            raise

    def update_learning_rate(self, new_learning_rate):
        try:
            self.learning_rate = new_learning_rate
            utils.update_config('learning_rate', new_learning_rate, source='runtime', silence=False)

            utils.xprint("Re")
            self.compute_update()

            utils.xprint("Re")
            self.compile()

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
                sampler.map_to_grid(grid_system=GridSystem(utils.get_config('grain_grid')))
            # Devide into train set & test set
            trainset = utils.get_config('trainset')
            trainset = int(trainset * sampler.length) if trainset < 1 else trainset
            sampler_trainset = Sampler.clip(sampler, indices=(0, trainset))
            sampler_testset = Sampler.clip(sampler, indices=(sampler_trainset.length, None))
            utils.xprint("Use %d samples as train set & %d samples as test set."
                         % (sampler_trainset.length, sampler_testset.length), newline=True)

            # Define the model
            model = SocialLSTM(node_identifiers=sampler.node_identifiers, motion_range=sampler.motion_range)

            try:
                # Import previously pickled parameters if requested
                file_unpickle = utils.get_config('file_unpickle') if utils.has_config('file_unpickle') else None
                params_unpickled = utils.filer.load_from_file(file_unpickle) if file_unpickle is not None else None

                # Build & compile the model
                outputs_var, params_var = model.build_network(params=params_unpickled)
                predictions_var = model.compute_prediction()
                loss_var = model.compute_loss()
                deviations_var = model.compute_deviation()
                updates_var = model.compute_update()
                func_predict, func_compare, func_train = model.compile()

                peek_e, peeks_hid, peek_outputs, peek_params, peek_probs = model.get_peeks()

                if params_unpickled is not None:
                    model.tryout(sampler_testset)

                utils.get_rootlogger().register("train", columns=["identifier", "loss-by-epoch", "deviation-by-epoch"])

                train_losses = numpy.zeros((0,))
                train_deviations = numpy.zeros((0,))

                # Do training
                try:

                    while True:
                        _losses, _deviations = model.train(sampler_trainset, num_epoch=utils.get_config('tryout_frequency'))
                        train_losses = numpy.append(train_losses, _losses)
                        train_deviations = numpy.append(train_deviations, _deviations)
                        if model.stop:
                            break

                        tryout_deviations = model.tryout(sampler_testset)

                        if model.entry_epoch >= model.num_epoch:

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
                            else:
                                break
                        else:
                            continue

                except Exception, e:
                    utils.handle(e)
                else:
                    utils.get_rootlogger().log({"identifier": utils.get_sublogger().identifier,
                                                "loss-by-epoch": '%s\n' % utils.format_var(train_losses, detail=True),
                                                "deviation-by-epoch": '%s\n' % utils.format_var(train_deviations,
                                                                                                detail=True)},
                                               name="train")

                finally:
                    model.export_history()

            except:
                raise
            finally:
                model.complete()
                utils.get_sublogger().log_config()

            def test_importing():
                _model = SocialLSTM(node_identifiers=sampler.node_identifiers, motion_range=sampler.motion_range)
                _model.build_network()
                _model.import_params()
                _model.compute_prediction()
                _model.compute_loss()
                _model.compute_deviation()
                _model.compute_update()
                _model.compile()

                # test_importing()

        except Exception, e:
            raise
