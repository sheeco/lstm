# coding:utf-8

import numpy

import theano
import theano.tensor as T
from theano.tensor.sharedvar import TensorSharedVariable
import lasagne as L

import utils
from sampler import Sampler

__all__ = [
    'SharedLSTM'
]


class SharedLSTM:
    train_schemes = ['rmsprop',
                     'adagrad',
                     'momentum',
                     'nesterov']

    loss_schemes = ['sum',
                    'mean']

    def __init__(self, sampler=None, motion_range=None, inputs=None, targets=None, dimension_embed_layer=None,
                 dimension_hidden_layer=None, grad_clip=None, num_epoch=None, loss_scheme=None, train_scheme=None,
                 adaptive_learning_rate=None, learning_rate=None, rho=None, epsilon=None, momentum=None):
        try:
            if __debug__:
                theano.config.exception_verbosity = 'high'
                theano.config.optimizer = 'fast_compile'

            # Sampler related variables

            if sampler is None:
                sampler = Sampler()
                sampler.pan_to_positive()
            self.sampler = sampler
            self.sampler.reset_entry()
            self.num_node = self.sampler.num_node
            self.nodes = self.sampler.node_identifiers
            if motion_range is None:
                self.motion_range = self.sampler.motion_range
            else:
                self.motion_range = motion_range

            # Variables defining the network
            self.dimension_sample = self.sampler.dimension_sample
            self.length_sequence_input = self.sampler.length_sequence_input
            self.length_sequence_output = self.sampler.length_sequence_output
            self.size_batch = self.sampler.size_batch
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
            self.loss_scheme = loss_scheme if loss_scheme is not None else utils.get_config('loss_scheme')
            if self.loss_scheme not in SharedLSTM.loss_schemes:
                raise ValueError("Unknown loss scheme '%s'. Must be among %s." % (self.loss_scheme, self.loss_schemes))

            self.train_scheme = train_scheme if train_scheme is not None else utils.get_config('train_scheme')
            if self.train_scheme not in SharedLSTM.train_schemes:
                raise ValueError(
                    "Unknown training scheme '%s'. Must be among %s." % (self.train_scheme, self.train_schemes))

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
            self.initial_param_values = None  # list of ndarrays, stored for possible parameter restoration
            self.network_history = []  # 2d list (iepoch, ibatch) of dict, data flow though the network

            # Theano function objects

            self.func_predict = None
            self.func_compare = None
            self.func_train = None

            self.check_embed = None
            self.checks_hid = []
            self.check_outputs = None
            self.check_params = None
            self.check_probs = None

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

            self.scaled_deviation = True

            self.w_deviations = L.init.Uniform(std=0.1, mean=(100. / self.dimension_hidden_layers[1] / self.num_node))
            self.b_deviations = L.init.Constant(0.)
            if self.scaled_deviation:
                self.f_deviations = L.nonlinearities.sigmoid
            else:
                self.f_deviations = L.nonlinearities.softplus

            self.scaled_correlation = True

            # self.w_correlation = L.init.Uniform(std=0.0005, mean=0.)
            self.w_correlation = L.init.Uniform(std=0., mean=0.)
            self.b_correlation = L.init.Constant(0.)
            if self.scaled_correlation:
                self.f_correlation = SharedLSTM.scaled_tanh
            else:
                self.f_correlation = SharedLSTM.safe_tanh

            # Prepare for logging

            self.root_logger = utils.get_rootlogger()
            self.sub_logger = utils.get_sublogger()

            self.sub_logger.register("training", columns=['epoch', 'batch', 'loss', 'deviations'])

            columns_prediction = ['epoch', 'batch', 'sample', 'instant'] + self.nodes
            self.sub_logger.register("prediction", columns=columns_prediction)

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
            return SharedLSTM.scale(y, beta)
            # return T.clip(y, -beta, beta)
        except:
            raise

    @staticmethod
    def safe_tanh(x, beta=.9):

        try:
            y = T.tanh(x)
            return SharedLSTM.scale(y, beta)
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

            utils.xprint('Building shared LSTM network ... ')

            # IN = [(sec, x, y)]
            layer_in = L.layers.InputLayer(name="input-layer", input_var=self.inputs,
                                           shape=(self.num_node, self.size_batch, self.length_sequence_input,
                                                  self.dimension_sample))
            # e = f_e(IN; W_e, b_e)
            layer_e = L.layers.DenseLayer(layer_in, name="e-layer", num_units=self.dimension_embed_layer, W=self.w_e,
                                          b=self.b_e,
                                          nonlinearity=self.f_e, num_leading_axes=3)
            assert utils.match(layer_e.output_shape,
                               (self.num_node, self.size_batch, self.length_sequence_input, self.dimension_embed_layer))

            layers_in_lstms = []
            for inode in xrange(0, self.num_node):
                layers_in_lstms += [L.layers.SliceLayer(layer_e, indices=inode, axis=0)]
            assert all(utils.match(ilayer_in_lstm.output_shape,
                                   (self.size_batch, self.length_sequence_input, self.dimension_embed_layer))
                       for ilayer_in_lstm in layers_in_lstms)

            n_hid = self.dimension_hidden_layers[0]
            dim_hid = self.dimension_hidden_layers[1]

            # Create the 1st LSTM layer for the 1st node
            layer_lstm_0 = L.layers.LSTMLayer(layers_in_lstms[0], dim_hid, name="LSTM-%d-%d" % (1, 1),
                                              nonlinearity=self.f_lstm_hid,
                                              ingate=L.layers.Gate(W_in=self.w_lstm_in,
                                                                   W_hid=self.w_lstm_hid,
                                                                   W_cell=self.w_lstm_hid,
                                                                   b=self.b_lstm),
                                              forgetgate=L.layers.Gate(W_in=self.w_lstm_in,
                                                                       W_hid=self.w_lstm_hid,
                                                                       W_cell=self.w_lstm_hid,
                                                                       b=self.b_lstm),
                                              cell=L.layers.Gate(W_cell=None, nonlinearity=self.f_lstm_cell),
                                              outgate=L.layers.Gate(W_in=self.w_lstm_in,
                                                                    W_hid=self.w_lstm_hid,
                                                                    W_cell=self.w_lstm_hid,
                                                                    b=self.b_lstm),
                                              hid_init=self.init_lstm_hid, cell_init=self.init_lstm_cell,
                                              only_return_final=False)
            assert utils.match(layer_lstm_0.output_shape, (self.size_batch, self.length_sequence_input, dim_hid))

            layers_lstm = [layer_lstm_0]

            # Create params sharing LSTMs for the rest (n - 1) nodes,
            # which have params exactly the same as LSTM-1-1
            for inode in xrange(1, self.num_node):
                layers_lstm += [
                    L.layers.LSTMLayer(layers_in_lstms[inode], dim_hid,
                                       name="LSTM-%d-%d" % (1, inode + 1),
                                       grad_clipping=self.grad_clip,
                                       nonlinearity=self.f_lstm_hid, hid_init=self.init_lstm_hid,
                                       cell_init=self.init_lstm_cell, only_return_final=False,
                                       ingate=L.layers.Gate(W_in=layer_lstm_0.W_in_to_ingate,
                                                            W_hid=layer_lstm_0.W_hid_to_ingate,
                                                            W_cell=layer_lstm_0.W_cell_to_ingate,
                                                            b=layer_lstm_0.b_ingate),
                                       outgate=L.layers.Gate(W_in=layer_lstm_0.W_in_to_outgate,
                                                             W_hid=layer_lstm_0.W_hid_to_outgate,
                                                             W_cell=layer_lstm_0.W_cell_to_outgate,
                                                             b=layer_lstm_0.b_outgate),
                                       forgetgate=L.layers.Gate(W_in=layer_lstm_0.W_in_to_forgetgate,
                                                                W_hid=layer_lstm_0.W_hid_to_forgetgate,
                                                                W_cell=layer_lstm_0.W_cell_to_forgetgate,
                                                                b=layer_lstm_0.b_forgetgate),
                                       cell=L.layers.Gate(W_in=layer_lstm_0.W_in_to_cell,
                                                          W_hid=layer_lstm_0.W_hid_to_cell,
                                                          W_cell=None,
                                                          b=layer_lstm_0.b_cell,
                                                          nonlinearity=self.f_lstm_cell
                                                          ))]

            layers_shuffled = []
            for inode in xrange(self.num_node):
                layers_shuffled += [L.layers.DimshuffleLayer(layers_lstm[inode], pattern=('x', 0, 1, 2))]
            layer_concated_lstms = L.layers.ConcatLayer(layers_shuffled, axis=0)
            layers_hid = [layer_concated_lstms]

            # Create more layers
            for i_hid in xrange(1, n_hid):
                layers_lstm[0] = L.layers.LSTMLayer(layers_lstm[0], dim_hid, name="LSTM-%d-%d" % (i_hid + 1, 1),
                                                    nonlinearity=self.f_lstm_hid,
                                                    ingate=L.layers.Gate(W_in=self.w_lstm_in,
                                                                         W_hid=self.w_lstm_hid,
                                                                         W_cell=self.w_lstm_hid,
                                                                         b=self.b_lstm),
                                                    forgetgate=L.layers.Gate(W_in=self.w_lstm_in,
                                                                             W_hid=self.w_lstm_hid,
                                                                             W_cell=self.w_lstm_hid,
                                                                             b=self.b_lstm),
                                                    cell=L.layers.Gate(W_cell=None, nonlinearity=self.f_lstm_cell),
                                                    outgate=L.layers.Gate(W_in=self.w_lstm_in,
                                                                          W_hid=self.w_lstm_hid,
                                                                          W_cell=self.w_lstm_hid,
                                                                          b=self.b_lstm),
                                                    hid_init=self.init_lstm_hid, cell_init=self.init_lstm_cell,
                                                    only_return_final=False)
                layer_lstm_0 = layers_lstm[0]
                assert utils.match(layer_lstm_0.output_shape, (self.size_batch, self.length_sequence_input, dim_hid))

                for inode in xrange(1, self.num_node):
                    layers_lstm[inode] = L.layers.LSTMLayer(layers_lstm[inode], dim_hid,
                                                            name="LSTM-%d-%d" % (i_hid + 1, inode + 1),
                                                            grad_clipping=self.grad_clip,
                                                            nonlinearity=self.f_lstm_hid, hid_init=self.init_lstm_hid,
                                                            cell_init=self.init_lstm_cell, only_return_final=False,
                                                            ingate=L.layers.Gate(W_in=layer_lstm_0.W_in_to_ingate,
                                                                                 W_hid=layer_lstm_0.W_hid_to_ingate,
                                                                                 W_cell=layer_lstm_0.W_cell_to_ingate,
                                                                                 b=layer_lstm_0.b_ingate),
                                                            outgate=L.layers.Gate(W_in=layer_lstm_0.W_in_to_outgate,
                                                                                  W_hid=layer_lstm_0.W_hid_to_outgate,
                                                                                  W_cell=layer_lstm_0.W_cell_to_outgate,
                                                                                  b=layer_lstm_0.b_outgate),
                                                            forgetgate=L.layers.Gate(
                                                                W_in=layer_lstm_0.W_in_to_forgetgate,
                                                                W_hid=layer_lstm_0.W_hid_to_forgetgate,
                                                                W_cell=layer_lstm_0.W_cell_to_forgetgate,
                                                                b=layer_lstm_0.b_forgetgate),
                                                            cell=L.layers.Gate(W_in=layer_lstm_0.W_in_to_cell,
                                                                               W_hid=layer_lstm_0.W_hid_to_cell,
                                                                               W_cell=None,
                                                                               b=layer_lstm_0.b_cell,
                                                                               nonlinearity=self.f_lstm_cell
                                                                               ))

                layers_shuffled = []
                for inode in xrange(self.num_node):
                    layers_shuffled += [L.layers.DimshuffleLayer(layers_lstm[inode], pattern=('x', 0, 1, 2))]
                layer_concated_lstms = L.layers.ConcatLayer(layers_shuffled, axis=0)
                assert utils.match(layer_concated_lstms.output_shape,
                                   (self.num_node, self.size_batch, self.length_sequence_input, dim_hid))
                layers_hid += [layer_concated_lstms]

            layer_last_hid = layers_hid[-1]
            layer_means = L.layers.DenseLayer(layer_last_hid, name="means-layer", num_units=2, W=self.w_means,
                                              b=self.b_means, nonlinearity=self.f_means, num_leading_axes=3)
            layer_deviations = L.layers.DenseLayer(layer_last_hid, name="deviations-layer", num_units=2,
                                                   W=self.w_deviations, b=self.b_deviations,
                                                   nonlinearity=self.f_deviations, num_leading_axes=3)
            layer_correlation = L.layers.DenseLayer(layer_last_hid, name="correlation-layer", num_units=1,
                                                    W=self.w_correlation, b=self.b_correlation,
                                                    nonlinearity=self.f_correlation, num_leading_axes=3)
            layer_distribution = L.layers.ConcatLayer([layer_means, layer_deviations, layer_correlation], axis=-1)

            assert utils.match(layer_distribution.output_shape,
                               (self.num_node, self.size_batch, self.length_sequence_input, 5))

            # Slice x-length sequence from the last, according to <length_sequence_output>
            layer_out = L.layers.SliceLayer(layer_distribution, indices=slice(-self.length_sequence_output, None),
                                            axis=2)
            assert utils.match(layer_out.output_shape,
                               (self.num_node, self.size_batch, self.length_sequence_output, 5))

            self.embed = L.layers.get_output(layer_e)
            for ilayer in layers_hid:
                self.hids += [L.layers.get_output(ilayer)]

            self.network = layer_out
            self.outputs = L.layers.get_output(layer_out)
            self.params_all = L.layers.get_all_params(layer_out)
            self.params_trainable = L.layers.get_all_params(layer_out, trainable=True)

            def get_names_for_params(list_params):
                utils.assertor.assert_type(list_params, list)
                param_keys = []
                for param in list_params:
                    # utils.assertor.assert_type(param, TensorSharedVariable)
                    name = param.name
                    param_keys += [name]
                return param_keys

            self.param_names = get_names_for_params(self.params_all)
            # Save initial values of params for possible future restoration
            self.initial_param_values = L.layers.get_all_param_values(layer_out)

            # Assign saved parameter values to the built network
            if params is not None:
                utils.xprint('Importing given parameters ... ')
                self.set_params(params)

            utils.xprint('done in %s.' % timer.stop(), newline=True)
            return self.outputs, self.params_all

        except:
            raise

    def build_decoder(self):
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
        NNL Loss for Training.
        Build computation graph from `predictions`, `targets` to `loss`.
        :return: `loss`
        """
        try:
            utils.assertor.assert_not_none(self.outputs, "Must build the network first.")
            utils.assertor.assert_not_none(self.predictions, "Must build the decoder first.")

            timer = utils.Timer()
            utils.xprint('Computing loss ... ')

            # Remove time column
            facts = self.targets[:, :, :, 1:3]
            shape_facts = facts.shape
            shape_stacked_facts = (shape_facts[0] * shape_facts[1] * shape_facts[2], shape_facts[3])

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
                deviations = distribution[2:4]
                correlation = distribution[5]
                target = fact_mat[idx, :]
                prob = SharedLSTM.bivar_norm(target[0], target[1], means[0], means[1], deviations[0], deviations[1],
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
                prob = SharedLSTM.bivar_norm(target[0], target[1], means[0], means[1], deviations[0], deviations[1],
                                             correlation)

                # Do something with the slice here.

                return prob

            # Make a vector containing the start idx of every slice
            indices = T.arange(facts.shape[0])

            if self.scaled_deviation:
                motion_range = T.constant(self.motion_range[1] - self.motion_range[0])
                probs, updates_loss = theano.scan(fn=step_loss_scaled, sequences=[indices],
                                                  non_sequences=[distributions, facts, motion_range])
            else:
                probs, updates_loss = theano.scan(fn=step_loss, sequences=[indices],
                                                  non_sequences=[distributions, facts])

            # Normal Negative Log-likelihood
            nnls = T.neg(T.log(probs))

            # Use either sum or mean for loss
            loss = None
            if self.loss_scheme == 'sum':
                loss = T.sum(nnls)
            elif self.loss_scheme == 'mean':
                loss = T.mean(nnls)
            else:
                raise ValueError("No definition found for loss scheme '%s'." % self.loss_scheme)

            utils.assertor.assert_not_none(loss, "Computation of loss has failed.")
            self.probabilities = probs
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
            utils.assertor.assert_not_none(self.predictions, "Must build the decoder first.")

            timer = utils.Timer()
            utils.xprint('Building observer ... ')

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
            utils.assertor.assert_not_none(self.predictions, "Must build the decoder first.")
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
            utils.assertor.assert_not_none(self.predictions, "Must build the decoder first.")
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
            Compile checking functions for debugging
            """

            self.check_embed = theano.function([self.inputs], self.embed, allow_input_downcast=True)
            for ihid in self.hids:
                self.checks_hid += [theano.function([self.inputs], ihid, allow_input_downcast=True)]
            self.check_outputs = theano.function([self.inputs], self.outputs, allow_input_downcast=True)
            self.check_params = theano.function([], self.params_all, allow_input_downcast=True)
            self.check_probs = theano.function([self.inputs, self.targets], self.probabilities,
                                               allow_input_downcast=True)

            utils.xprint('done in %s.' % timer.stop(), newline=True)
            return self.func_predict, self.func_compare, self.func_train

        except:
            raise

    def get_checks(self):

        return self.check_embed, self.checks_hid, self.check_outputs, self.check_params, self.check_probs

    def train(self):
        """

        :return: Two <ndarray> containing average loss & deviation of each epoch.
        """
        utils.assertor.assert_not_none(self.outputs, "Must build the network first.")
        utils.assertor.assert_not_none(self.predictions, "Must build the decoder first.")
        utils.assertor.assert_not_none(self.loss, "Must compute the loss first.")
        utils.assertor.assert_not_none(self.deviations, "Must compute the deviation first.")
        if any(func is None for func in [self.func_predict, self.func_compare, self.func_train]):
            raise RuntimeError("Must compile the functions first.")

        invalid = True
        try:
            while invalid:
                # start of single training try
                try:
                    utils.xprint('Training ... ', newline=True)
                    timer = utils.Timer()

                    loss_epoch = numpy.zeros((0,))
                    deviation_epoch = numpy.zeros((0,))
                    stop = False  # Whether to stop and exit
                    completed = None  # Whether a batch has got proceeded completely

                    # for iepoch in range(self.num_epoch):
                    iepoch = 0
                    while True:
                        # start of single epoch
                        utils.xprint('  Epoch %d ... ' % iepoch, newline=True)
                        self.network_history.append([])
                        loss = None
                        deviations = None
                        loss_batch = numpy.zeros((0,))
                        deviation_batch = numpy.zeros((0,))
                        ibatch = 0
                        while True:
                            # start of single batch
                            try:

                                # retrieve the next batch for nodes
                                # only if the previous batch is completed
                                # else, redo the previous batch
                                if completed is None \
                                        or completed:
                                    completed = False
                                    instants, inputs, targets = self.sampler.load_batch(with_target=True)
                                if inputs is None:
                                    if ibatch == 0:
                                        raise RuntimeError("Only %d sample pairs are found, "
                                                           "not enough for one single batch of size %d."
                                                           % (self.sampler.length, self.size_batch))

                                    self.sampler.reset_entry()
                                    completed = None
                                    break

                                def check_netflow():
                                    embed = self.check_embed(inputs)
                                    hids = []
                                    for ihid in xrange(self.dimension_hidden_layers[0]):
                                        check_hid = self.checks_hid[ihid]
                                        hids += [check_hid(inputs)]
                                    netout = self.check_outputs(inputs)

                                    dict_netflow = {'embed': embed, 'hiddens': hids, 'netout': netout}
                                    return dict_netflow

                                utils.xprint('    Batch %d ... ' % ibatch)

                                # record params, flow of data & probabilities to network history BEFORE training
                                params = self.check_params()

                                netflow = check_netflow()
                                probs = self.check_probs(inputs, targets)
                                # note that record [i, j] contains variable values BEFORE this training
                                self.network_history[iepoch].append({'params': params, 'netflow': netflow, 'probs': probs})

                                predictions = self.func_predict(inputs)
                                deviations = self.func_compare(inputs, targets)

                                # Validate loss

                                loss = self.func_train(inputs, targets)
                                try:
                                    utils.assertor.assert_finite(loss, 'loss')

                                except AssertionError, e:
                                    raise utils.InvalidTrainError("Get loss of 'inf'.",
                                                                  details="Network output:\n"
                                                                          "%s" % netflow['netout'])

                                # Validate params after training

                                new_params = self.check_params()
                                try:
                                    utils.assertor.assert_finite(new_params, 'params')

                                except AssertionError, e:
                                    raise utils.InvalidTrainError(
                                        "Get parameters containing 'nan' or 'inf' after training.",
                                        details="Parameters:\n"
                                                "%s" % new_params)
                                else:
                                    # consider successful if training is done and successful
                                    completed = True

                            except KeyboardInterrupt, e:
                                utils.xprint('', newline=True)
                                timer.pause()
                                stop = utils.ask("Stop and exit?", code_quit=None, interpretor=utils.interpret_confirm)
                                timer.resume()
                                if stop:
                                    # means n complete epochs
                                    utils.update_config('num_epoch', iepoch, 'runtime', silence=False)
                                    break
                                else:
                                    continue

                            finally:
                                if completed:

                                    # Log predictions, targets & deviations

                                    def log_predictions():
                                        size_this_batch = len(instants)
                                        for isample in xrange(0, size_this_batch):

                                            dict_content = {'epoch': iepoch, 'batch': ibatch, 'sample': isample}

                                            for iseq in xrange(0, self.length_sequence_output):
                                                # index in [-n, -1]
                                                dict_content['instant'] = instants[
                                                    isample, iseq - self.length_sequence_output]
                                                for inode in xrange(0, self.num_node):
                                                    # [x, y]
                                                    deviation_ = deviations[inode, isample, iseq]
                                                    prediction_ = predictions[inode, isample, iseq]
                                                    target_ = targets[inode, isample, iseq]
                                                    dict_content[self.nodes[inode]] = "(%.1f, [%.1f, %.1f], [%.1f, %.1f])" \
                                                                                      % (deviation_, prediction_[0],
                                                                                         prediction_[1],
                                                                                         target_[-2], target_[-1])

                                                self.sub_logger.log(dict_content, name="prediction")

                                    log_predictions()

                                    # Log loss together with brief deviation info

                                    loss_batch = numpy.append(loss_batch, loss)
                                    deviation_batch = numpy.append(deviation_batch, numpy.mean(deviations))

                                    self.sub_logger.log({'epoch': iepoch, 'batch': ibatch,
                                                         'loss': utils.format_var(float(loss)),
                                                         'deviations': utils.format_var(deviations)},
                                                        name="training")

                                    utils.xprint('%s; %s'
                                                 % (utils.format_var(float(loss), name='loss'),
                                                    utils.format_var(deviations, name='deviations')),
                                                 newline=True)

                                    ibatch += 1

                            # end of single batch
                        # end of single epoch

                        loss_epoch = numpy.append(loss_epoch, numpy.mean(loss_batch))
                        deviation_epoch = numpy.append(deviation_epoch, numpy.mean(deviation_batch))

                        iepoch += 1
                        utils.update_config('num_epoch', iepoch, 'runtime')

                        if stop:
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

                    # end of all epochs
                    invalid = False

                # end of single training try
                except utils.InvalidTrainError, e:
                    # Update learning rate & Retrain

                    if self.adaptive_learning_rate is not None:
                        utils.warn(e.message)
                        new_learning_rate = self.learning_rate * self.adaptive_learning_rate
                        self.update_learning_rate(new_learning_rate)

                        # Reinitialize related variables
                        self.network_history = []
                        self.sampler.reset_entry()
                        continue

                    else:
                        raise
        except:
            raise

        else:
            utils.xprint('Done in %s.' % timer.stop(), newline=True)
            return loss_epoch, deviation_epoch

        finally:
            self.export_params()
            self.sub_logger.log_config()

    def export_params(self, path=None):
        try:
            FILENAME_EXPORT = 'params.pkl'

            timer = utils.Timer()
            utils.assertor.assert_not_none(self.params_all, "Must build the network first.")

            if path is None:
                path = utils.filer.format_subpath(self.sub_logger.log_path, FILENAME_EXPORT)

            utils.xprint("\nExporting parameters to '%s' ...  " % path)
            utils.update_config('path_pickle', path, 'runtime', tags=['path'])

            if self.check_params is None:
                self.check_params = theano.function([], self.params_all, allow_input_downcast=True)

            params_all = self.check_params()

            utils.filer.dump_to_file(params_all, path)

            utils.xprint('done in %s.' % timer.stop(), newline=True)
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
        self.sub_logger.complete()

    @staticmethod
    def test():

        try:
            utils.get_rootlogger().register("loss", columns=["identifier", "loss-by-epoch", "deviation-by-epoch"])

            # Select certain nodes if requested
            nodes = utils.get_config('nodes') if utils.has_config('nodes') else None
            nodes = utils.get_config('num_node') if nodes is None and utils.has_config('num_node') else nodes

            # Build sampler
            sampler = Sampler(nodes=nodes, keep_positive=True)
            half = Sampler.clip(sampler, indices=(sampler.length / 2))

            # Define the model
            model = SharedLSTM(sampler=half, motion_range=sampler.motion_range)

            try:
                # Import previously pickled parameters if requested
                path_unpickle = utils.get_config('path_unpickle') if utils.has_config('path_unpickle') else None
                params_unpickled = utils.filer.load_from_file(path_unpickle) if path_unpickle is not None else None
                if params_unpickled is not None:
                    utils.get_sublogger().log_file(path_unpickle, rename='params-imported.pkl')

                # Build & compile the model
                outputs_var, params_var = model.build_network(params=params_unpickled)
                predictions_var = model.build_decoder()
                loss_var = model.compute_loss()
                deviations_var = model.compute_deviation()
                updates_var = model.compute_update()
                func_predict, func_compare, func_train = model.compile()

                check_e, checks_hid, check_outputs, check_params, check_probs = model.get_checks()

                utils.get_rootlogger().register("loss", columns=["identifier", "loss-by-epoch", "deviation-by-epoch"])

                # Do training
                loss, deviations = model.train()

                def test_importing():
                    model3 = SharedLSTM(sampler=half, motion_range=sampler.motion_range)
                    model3.build_network()
                    model3.import_params()
                    model3.build_decoder()
                    model3.compute_loss()
                    model3.compute_deviation()
                    model3.compute_update()
                    model3.compile()

                # test_importing()

                utils.get_rootlogger().log({"identifier": utils.get_sublogger().identifier,
                                            "loss-by-epoch": '%s\n' % utils.format_var(loss, detail=True),
                                            "deviation-by-epoch": '%s\n' % utils.format_var(deviations, detail=True)},
                                           name="loss")
            except:
                raise
            finally:
                model.complete()

        except Exception, e:
            raise
