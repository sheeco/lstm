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

            self.loss = None  # numpy float, loss computed for single batch
            self.predictions = None  # numpy ndarray, predictions of single batch
            self.deviations = None  # numpy ndarray, euclidean distances between predictions & targets
            self.probabilities = None  # numpy ndarray, binorm probabilities before computing into NNL

            self.param_names = []  # list of str, names of all the parameters
            self.param_values = None  # list of ndarrays, stored for debugging or exporting
            self.initial_param_values = None  # list of ndarrays, stored for possible parameter restoration

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

            columns_prediction = ['epoch', 'batch', 'sample', 'instant'] + self.node_identifiers
            self.logger.register("training-sample", columns=columns_prediction)
            self.logger.register("training-batch", columns=['epoch', 'batch', 'loss',
                                                            'mean-deviation', 'min-deviation', 'max-deviation'])

            self.logger.register("training-epoch", columns=["epoch", "mean-loss",
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
                                           shape=(self.num_node, self.size_batch, self.length_sequence_input,
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
                               (self.num_node, self.size_batch, self.length_sequence_input, self.dimension_embed_layer))

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
            # todo add the sharing of social tensor H maybe

            assert all(utils.match(_each_input_hidden.output_shape,
                                   (self.size_batch, self.length_sequence_input, None))
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

                    assert utils.match(_lstm.output_shape, (self.size_batch, self.length_sequence_input, dim_hid))
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

                    assert utils.match(_lstm.output_shape, (self.size_batch, self.length_sequence_input, dim_hid))
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
                               (self.num_node, self.size_batch, self.length_sequence_input, dim_hid))

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
                                   (self.num_node, self.size_batch, self.length_sequence_input, 5))
                layer_decoded = layer_distribution

            elif self.decode_scheme == 'euclidean':
                assert utils.match(layer_means.output_shape,
                                   (self.num_node, self.size_batch, self.length_sequence_input, 2))
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
            # Save initial values of params for possible future restoration
            self.initial_param_values = L.layers.get_all_param_values(layer_out)

            """
            Import external paratemers if given
            """

            # Assign saved parameter values to the built network
            if params is not None:
                utils.xprint('Importing given parameters ... ')
                self.set_params(params)

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

    def train(self, sampler):
        """

        :return: Two <ndarray> containing average loss & deviation of each epoch.
        """
        utils.assertor.assert_not_none(self.outputs, "Must build the network first.")
        utils.assertor.assert_not_none(self.predictions, "Must compute the prediction first.")
        utils.assertor.assert_not_none(self.loss, "Must compute the loss first.")
        utils.assertor.assert_not_none(self.deviations, "Must compute the deviation first.")
        for _func in (self.func_predict, self.func_compare, self.func_train):
            utils.assertor.assert_not_none(_func, "Must compile the functions first.")

        done_training = False
        try:
            while not done_training:
                # start of single training try
                try:
                    utils.xprint('Training ... ', newline=True)
                    timer = utils.Timer()

                    losses_by_epoch = numpy.zeros((0,))
                    deviations_by_epoch = numpy.zeros((0,))
                    params = None
                    do_stop = False  # Whether to stop and exit

                    # for iepoch in range(self.num_epoch):
                    iepoch = 0
                    while True:
                        # start of single epoch
                        done_epoch = False  # Whether an epoch has got finished properly
                        utils.xprint('  Epoch %d ... ' % iepoch, newline=True)
                        if self.network_history is not None:
                            self.network_history.append([])
                        loss = None
                        deviations = None
                        losses_by_batch = numpy.zeros((0,))
                        deviations_by_batch = numpy.zeros((0,))
                        done_batch = None  # Whether a batch has got finished properly

                        ibatch = 0
                        while True:
                            # start of single batch
                            try:
                                # sleep a bit to catch KeyboardInterrupt
                                utils.sleep(0.001)

                                # retrieve the next batch for nodes
                                # only if the previous batch is completed
                                # else, redo the previous batch
                                if done_batch is None \
                                        or done_batch:
                                    done_batch = False
                                    instants_input, inputs, instants_target, targets = sampler.load_batch(with_target=True)
                                if inputs is None:
                                    if ibatch == 0:
                                        raise RuntimeError("Only %d sample pairs are found, "
                                                           "not enough for one single batch of size %d."
                                                           % (sampler.length, self.size_batch))

                                    sampler.reset_entry()
                                    done_batch = None
                                    done_epoch = True
                                    break

                                utils.xprint('    Batch %d ... ' % ibatch)

                                if params is None:
                                    params = self.peek_params()
                                self.param_values = params

                                # Record params, flow of data & probabilities to network history BEFORE training

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
                                    self.network_history[-1].append({'params': params,
                                                                     'netflow': _netflow,
                                                                     'probs': _probs})

                                    # throw away overdue history
                                    if self.limit_network_history is not None \
                                            and len(self.network_history) > self.limit_network_history:
                                        self.network_history = \
                                            self.network_history[-self.limit_network_history:-1]

                                predictions = self.func_predict(inputs)
                                deviations = self.func_compare(inputs, targets)

                                # Validate loss

                                loss = self.func_train(inputs, targets)
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
                                    params = new_params
                                    # consider successful if training is done and successful
                                    done_batch = True

                            except KeyboardInterrupt, e:

                                _menu = ('stop', 'continue', 'peek')
                                _abbr_menu = ('s', 'c', 'p')
                                _hint_menu = "0: (s)top & exit   1: (c)ontinue    2: (p)eek network output"

                                def interpret_menu(answer):
                                    try:
                                        if answer in _menu:
                                            return _menu.index(answer)
                                        elif answer in _abbr_menu:
                                            return _abbr_menu.index(answer)
                                        else:
                                            n = int(answer)
                                            if 0 <= answer < len(_menu):
                                                return n
                                            else:
                                                raise AssertionError("Choice out of scope.")
                                    except Exception, e:
                                        raise AssertionError(e.message)

                                utils.xprint('\n', newline=True)
                                timer.pause()
                                choice = utils.ask(_hint_menu, code_quit='q', interpretor=interpret_menu)
                                utils.xprint('', newline=True)
                                timer.resume()

                                while choice == _menu.index('peek'):
                                    _netout = self.peek_outputs(inputs)
                                    utils.xprint('Network Output:\n%s\n' % _netout, newline=True)

                                    # ask again after peeking
                                    utils.xprint('', newline=True)
                                    timer.pause()
                                    choice = utils.ask(_hint_menu, code_quit='q', interpretor=interpret_menu)
                                    utils.xprint('', newline=True)
                                    timer.resume()

                                if choice == _menu.index('stop'):
                                    do_stop = True
                                    # means n complete epochs
                                    utils.update_config('num_epoch', iepoch, 'runtime', silence=False)
                                    break
                                else:
                                    continue

                            finally:

                                if done_batch:

                                    # Make sure complete logging discarding KeyboardInterrupt
                                    _done_logging = False
                                    while not _done_logging:
                                        try:
                                            # Log [deviation, prediction, target] by each sample

                                            def log_by_sample():
                                                size_this_batch = len(instants_input)
                                                for isample in xrange(0, size_this_batch):

                                                    dict_content = {'epoch': iepoch, 'batch': ibatch, 'sample': isample}

                                                    for iseq in xrange(0, self.length_sequence_output):
                                                        # index in [-n, -1]
                                                        dict_content['instant'] = instants_target[
                                                            isample, iseq - self.length_sequence_output]
                                                        for inode in xrange(0, self.num_node):
                                                            # [x, y]
                                                            _deviation = deviations[inode, isample, iseq]
                                                            _prediction = predictions[inode, isample, iseq]
                                                            _target = targets[inode, isample, iseq, -2:]
                                                            dict_content[self.node_identifiers[inode]] = "(%.2f, %s, %s)" \
                                                                                                         % (_deviation, _prediction,
                                                                                                 _target)

                                                        self.logger.log(dict_content, name="training-sample")

                                            log_by_sample()

                                            losses_by_batch = numpy.append(losses_by_batch, loss)
                                            deviations_by_batch = numpy.append(deviations_by_batch, numpy.mean(deviations))

                                            # Print loss & deviation info to console
                                            utils.xprint('%s; %s'
                                                         % (utils.format_var(float(loss), name='loss'),
                                                            utils.format_var(deviations, name='deviations')),
                                                         newline=True)

                                            # Log [loss, mean-deviation, min-deviation, max-deviation] by each batch

                                            def log_by_batch():
                                                _peek_deviations_this_batch = utils.peek_matrix(deviations, formatted=True)

                                                self.logger.log({'epoch': iepoch, 'batch': ibatch,
                                                                 'loss': utils.format_var(float(loss)),
                                                                 'mean-deviation': _peek_deviations_this_batch['mean'],
                                                                 'min-deviation': _peek_deviations_this_batch['min'],
                                                                 'max-deviation': _peek_deviations_this_batch['max']},
                                                                name="training-batch")
                                            log_by_batch()

                                            ibatch += 1
                                            break
                                        except KeyboardInterrupt, e:
                                            pass
                                    pass  # end of while not _done_logging
                                else:  # skip logging if this batch is undone
                                    pass

                            pass  # end of single batch
                        pass  # end of single epoch

                        if done_epoch:
                            losses_by_epoch = numpy.append(losses_by_epoch, numpy.mean(losses_by_batch))
                            deviations_by_epoch = numpy.append(deviations_by_epoch, numpy.mean(deviations_by_batch))

                            _peek_losses_this_epoch = utils.peek_matrix(losses_by_batch, formatted=True)
                            _peek_deviations_this_epoch = utils.peek_matrix(deviations_by_batch, formatted=True)

                            # Print loss & deviation info to console
                            utils.xprint('  mean-loss: %s; mean-deviation: %s'
                                         % (_peek_losses_this_epoch['mean'], _peek_deviations_this_epoch['mean']),
                                         newline=True)

                            # Log [mean-loss, mean-deviation, min-deviation, max-deviation] by each epoch

                            def log_by_epoch():
                                self.logger.log({'epoch': iepoch,
                                                 'mean-loss': _peek_losses_this_epoch['mean'],
                                                 'mean-deviation': _peek_deviations_this_epoch['mean'],
                                                 'min-deviation': _peek_deviations_this_epoch['min'],
                                                 'max-deviation': _peek_deviations_this_epoch['max']},
                                                name="training-epoch")

                            log_by_epoch()

                            iepoch += 1
                        else:  # skip logging if this epoch is undone
                            pass

                        if do_stop:
                            break

                        elif iepoch >= self.num_epoch:

                            timer.pause()
                            more = utils.ask("Try more epochs?", code_quit=None, interpretor=utils.interpret_confirm)
                            timer.resume()

                            if more:

                                timer.pause()
                                num_more = utils.ask("How many?", interpretor=utils.interpret_positive_int)
                                timer.resume()

                                # quit means no more epochs
                                if num_more is not None \
                                        and num_more > 0:
                                    self.num_epoch += num_more
                                    utils.update_config('num_epoch', self.num_epoch, 'runtime', silence=False)
                                else:
                                    break
                            else:
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
                        if self.network_history is not None:
                            self.network_history = []

                        continue

                    else:
                        raise

                pass  # end of single training attempt
            pass  # end of while not done_training
        except:
            raise

        else:
            utils.xprint('Done in %s.' % timer.stop(), newline=True)
            return losses_by_epoch, deviations_by_epoch

    def export_history(self, path=None):
        try:
            if self.network_history is None \
                    or len(self.network_history) == 0:
                return

            FILENAME_EXPORT = 'history.pkl'

            if path is None:
                path = utils.filer.format_subpath(self.logger.log_path, FILENAME_EXPORT)

            utils.xprint("\nExporting recent network history to '%s' ...  " % path)

            utils.filer.dump_to_file(self.network_history, path)

            utils.xprint('done.', newline=True)
            return path

        except:
            raise

    def export_params(self, path=None):
        try:
            FILENAME_EXPORT = 'params.pkl'

            utils.assertor.assert_not_none(self.params_all, "Must build the network first.")

            if path is None:
                path = utils.filer.format_subpath(self.logger.log_path, FILENAME_EXPORT)

            utils.xprint("\nExporting parameters to '%s' ...  " % path)
            utils.update_config('path_pickle', path, 'runtime', tags=['path'])
            # last validated values during training
            if self.param_values is not None:
                params_all = self.param_values
            # initial values
            else:
                params_all = self.initial_param_values

            utils.filer.dump_to_file(params_all, path)

            utils.xprint('done.', newline=True)
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
            self.set_params(params_all)
            utils.xprint('done.', newline=True)

        except:
            raise

    def set_params(self, params):
        try:
            L.layers.set_all_param_values(self.network, params)
            # update initial param values to restore from
            self.initial_param_values = params

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
            self.set_params(self.initial_param_values)
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
            half = Sampler.clip(sampler, indices=(sampler.length / 2))

            # Define the model
            model = SocialLSTM(node_identifiers=sampler.node_identifiers, motion_range=sampler.motion_range)

            try:
                # Import previously pickled parameters if requested
                path_unpickle = utils.get_config('path_unpickle') if utils.has_config('path_unpickle') else None
                params_unpickled = utils.filer.load_from_file(path_unpickle) if path_unpickle is not None else None
                if params_unpickled is not None:
                    utils.get_sublogger().log_file(path_unpickle, rename='params-imported.pkl')

                # Build & compile the model
                outputs_var, params_var = model.build_network(params=params_unpickled)
                predictions_var = model.compute_prediction()
                loss_var = model.compute_loss()
                deviations_var = model.compute_deviation()
                updates_var = model.compute_update()
                func_predict, func_compare, func_train = model.compile()

                peek_e, peeks_hid, peek_outputs, peek_params, peek_probs = model.get_peeks()

                utils.get_rootlogger().register("training", columns=["identifier", "loss-by-epoch", "deviation-by-epoch"])

                # Do training
                try:
                    loss, deviations = model.train(sampler)
                except Exception, e:
                    utils.handle(e)
                else:
                    utils.get_rootlogger().log({"identifier": utils.get_sublogger().identifier,
                                                "loss-by-epoch": '%s\n' % utils.format_var(loss, detail=True),
                                                "deviation-by-epoch": '%s\n' % utils.format_var(deviations,
                                                                                                detail=True)},
                                               name="training")

                finally:
                    model.export_history()
                    model.export_params()

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
