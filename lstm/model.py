# coding:GBK

import numpy
import theano
import theano.tensor as T
from theano.compile.nanguardmode import NanGuardMode
from theano.tensor.sharedvar import TensorSharedVariable
import lasagne as L

import config
import utils
from sampler import *
import filer

__all__ = [
    'SharedLSTM'
]


# todo add debug info & assertion
# todo write config to .log
# todo read config from command line args


class SharedLSTM:
    def __init__(self, sampler=None, motion_range=None, inputs=None, targets=None, dimension_embed_layer=config.DIMENSION_EMBED_LAYER,
                 dimension_hidden_layers=config.DIMENSION_HIDDEN_LAYERS, grad_clip=config.GRAD_CLIP,
                 num_epoch=config.NUM_EPOCH,
                 learning_rate_rmsprop=config.LEARNING_RATE_RMSPROP, rho_rmsprop=config.RHO_RMSPROP,
                 epsilon_rmsprop=config.EPSILON_RMSPROP, logger=None):
        try:
            if sampler is None:
                sampler = Sampler()
                sampler.pan_to_positive()
            self.sampler = sampler
            self.sampler.reset_entry()
            self.num_node = self.sampler.num_node
            if motion_range is None:
                self.motion_range = self.sampler.motion_range
            else:
                self.motion_range = motion_range

            self.dimension_sample = self.sampler.dimension_sample
            self.length_sequence_input = self.sampler.length_sequence_input
            self.length_sequence_output = self.sampler.length_sequence_output
            self.size_batch = self.sampler.size_batch
            self.dimension_embed_layer = dimension_embed_layer

            utils.assert_type(dimension_hidden_layers, tuple)
            all(utils.assert_type(x, int) for x in dimension_hidden_layers)

            if len(dimension_hidden_layers) == 1:
                dimension_hidden_layers = (1,) + dimension_hidden_layers
            elif len(dimension_hidden_layers) == 2:
                pass
            else:
                raise ValueError("__init__ @ SharedLSTM: Expect len: 1~2 while getting %d instead.",
                                 len(dimension_hidden_layers))
            self.dimension_hidden_layers = dimension_hidden_layers
            self.grad_clip = grad_clip
            self.num_epoch = num_epoch

            self.learning_rate_rmsprop = learning_rate_rmsprop
            self.rho_rmsprop = rho_rmsprop
            self.epsilon_rmsprop = epsilon_rmsprop

            if inputs is None:
                inputs = T.tensor4("input_var", dtype='float32')
            if targets is None:
                targets = T.tensor4("target_var", dtype='float32')

            self.inputs = inputs
            self.targets = targets

            self.embed = None
            self.hids = []
            self.outputs = None
            self.network = None
            self.params_all = None
            self.params_trainable = None
            self.param_names = []
            self.updates = None
            self.predictions = None
            self.probabilities = None
            self.loss = None
            self.deviations = None

            self.network_history = {}

            self.func_predict = None
            self.func_compare = None
            self.func_train = None

            self.check_embedded = None
            self.checks_hid = []
            self.check_outputs = None
            self.check_params = None
            self.check_probs = None

            """
            Initialization Definitions
            """

            self.w_e = L.init.Uniform(std=0.005, mean=(1. / self.dimension_sample))
            # self.w_e = L.init.Uniform(range=(0., 1.))
            self.b_e = L.init.Constant(0.)
            self.f_e = None

            self.w_lstm_in = L.init.Uniform(std=0.005, mean=(1. / self.dimension_embed_layer))
            self.w_lstm_hid = L.init.Uniform(std=0.005, mean=(1. / self.dimension_hidden_layers[1]))
            self.w_lstm_cell = L.init.Uniform(std=0.005, mean=(1. / self.dimension_hidden_layers[1]))
            self.b_lstm = L.init.Constant(0.)
            # self.f_lstm_hid = L.nonlinearities.softplus
            # self.f_lstm_cell = L.nonlinearities.softplus
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

            if logger is None:
                logger = filer.Logger(identifier=utils.get_timestamp())
                logger.copy_config()
                logger.register_console()
            self.logger = logger
            self.logger.register("training", tags=['epoch', 'batch', 'loss', 'deviations'])

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
    def _check_bivar_norm(fact, distribution):
        """
        :param fact: [x1, x2]
        :param distribution: [mu1, mu2, sigma1, sigma2, rho]
        """
        try:
            _prob = SharedLSTM.bivar_norm(fact[0], fact[1], distribution[0], distribution[1], distribution[2],
                                          distribution[3],
                                          distribution[4])
            _val = _prob.eval()
            return _val
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
        try:
            timer = utils.Timer()

            utils.xprint('Building shared LSTM network ...', logger=self.logger)

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
            assert all(
                utils.match(ilayer_in_lstm.output_shape,
                            (self.size_batch, self.length_sequence_input, self.dimension_embed_layer)) for
                ilayer_in_lstm
                in layers_in_lstms)

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
            layer_means = L.layers.DenseLayer(layer_last_hid, name="means-layer", num_units=2, W=self.w_means, b=self.b_means,
                                              nonlinearity=self.f_means, num_leading_axes=3)
            layer_deviations = L.layers.DenseLayer(layer_last_hid, name="deviations-layer", num_units=2, W=self.w_deviations,
                                                   b=self.b_deviations,
                                                   nonlinearity=self.f_deviations, num_leading_axes=3)
            layer_correlation = L.layers.DenseLayer(layer_last_hid, name="correlation-layer", num_units=1,
                                                    W=self.w_correlation,
                                                    b=self.b_correlation,
                                                    nonlinearity=self.f_correlation, num_leading_axes=3)
            layer_distribution = L.layers.ConcatLayer([layer_means, layer_deviations, layer_correlation], axis=-1)

            assert utils.match(layer_distribution.output_shape,
                               (self.num_node, self.size_batch, self.length_sequence_input, 5))

            # Slice x-length sequence from the last, according to <length_sequence_output>
            layer_out = L.layers.SliceLayer(layer_distribution, indices=slice(-self.length_sequence_output, None),
                                            axis=2)
            assert utils.match(layer_distribution.output_shape,
                               (self.num_node, self.size_batch, self.length_sequence_output, 5))

            self.embed = L.layers.get_output(layer_e)
            for ilayer in layers_hid:
                self.hids += [L.layers.get_output(ilayer)]

            self.network = layer_out
            self.outputs = L.layers.get_output(layer_out)
            self.params_all = L.layers.get_all_params(layer_out)
            self.params_trainable = L.layers.get_all_params(layer_out, trainable=True)
            # self.params = L.layers.get_all_params(layer_out, unwrap_shared=True, trainable=True)

            def get_names_for_params(list_params):
                utils.assert_type(list_params, list)
                param_keys = []
                for param in list_params:
                    # utils.assert_type(param, TensorSharedVariable)
                    name = param.name
                    # ndim = param.ndim
                    param_keys += [name]
                return param_keys

            self.param_names = get_names_for_params(self.params_all)

            # Assign saved parameter values to the built network
            if params is not None:
                self.set_params(params)

            utils.xprint('Done in %s' % timer.stop(), newline=True, logger=self.logger)
            return self.outputs, self.params_all

        except:
            raise

    # todo extract decode() from compile()
    def compile(self):

        try:
            timer = utils.Timer()

            utils.xprint('Preparing ...', logger=self.logger)

            # Use mean(x, y) as predictions directly
            predictions = self.outputs[:, :, :, 0:2]
            # Remove time column
            facts = self.targets[:, :, :, 1:3]
            shape_facts = facts.shape
            shape_stacked_facts = (shape_facts[0] * shape_facts[1] * shape_facts[2], shape_facts[3])

            """
            Euclidean Error for Observation
            """

            # Elemwise differences
            differences = T.sub(predictions, facts)
            differences = T.reshape(differences, shape_stacked_facts)
            deviations = T.add(differences[:, 0] ** 2, differences[:, 1] ** 2) ** 0.5
            shape_deviations = (shape_facts[0], shape_facts[1], shape_facts[2])
            deviations = T.reshape(deviations, shape_deviations)

            """
            NNL Loss for Training
            """

            # Reshape for convenience
            facts = T.reshape(facts, shape_stacked_facts)
            shape_distributions = self.outputs.shape
            shape_stacked_distributions = (shape_distributions[0] * shape_distributions[1] * shape_distributions[2], shape_distributions[3])
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
                probs, updates_loss = theano.scan(fn=step_loss_scaled,
                                                  sequences=[indices],
                                                  non_sequences=[distributions, facts, motion_range])
            else:
                probs, updates_loss = theano.scan(fn=step_loss,
                                                  sequences=[indices],
                                                  non_sequences=[distributions, facts])

            # Normal Negative Log-likelihood
            nnls = T.neg(T.log(probs))
            # loss = T.sum(nnls)
            loss = T.mean(nnls)
            # loss = T.mean(deviations)

            utils.xprint('Done in %s' % timer.stop(), newline=True, logger=self.logger)
            timer.start()
            utils.xprint('Computing updates ...', logger=self.logger)

            # Compute RMSProp updates for training
            RMSPROP = L.updates.rmsprop(loss, self.params_trainable, learning_rate=self.learning_rate_rmsprop, rho=self.rho_rmsprop,
                                        epsilon=self.epsilon_rmsprop)
            updates = RMSPROP

            self.updates = updates
            self.predictions = predictions
            self.probabilities = probs
            self.loss = loss
            self.deviations = deviations

            utils.xprint('Done in %s' % timer.stop(), newline=True, logger=self.logger)
            timer.start()
            utils.xprint('Compiling functions ...', logger=self.logger)

            """
            Compile theano functions for prediction, observation & training
            """
            self.func_predict = theano.function([self.inputs], self.predictions, allow_input_downcast=True)
            self.func_compare = theano.function([self.inputs, self.targets], self.deviations, allow_input_downcast=True)
            self.func_train = theano.function([self.inputs, self.targets], self.loss, updates=updates,
                                              allow_input_downcast=True)
            # self.func_train = theano.function([self.inputs, self.targets], self.loss, updates=updates,
            #                                   allow_input_downcast=True,
            #                                   mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))

            """
            Compile checking functions for debugging
            """
            self.check_embedded = theano.function([self.inputs], self.embed, allow_input_downcast=True)
            for ihid in self.hids:
                self.checks_hid += [theano.function([self.inputs], ihid, allow_input_downcast=True)]
            self.check_outputs = theano.function([self.inputs], self.outputs, allow_input_downcast=True)
            self.check_params = theano.function([], self.params_all, allow_input_downcast=True)
            self.check_probs = theano.function([self.inputs, self.targets], self.probabilities, allow_input_downcast=True)

            utils.xprint('Done in %s' % timer.stop(), newline=True, logger=self.logger)
            return self.func_predict, self.func_compare, self.func_train

        except:
            raise

    def get_checks(self):

        return self.check_embedded, self.checks_hid, self.check_outputs, self.check_params, self.check_probs

    def train(self, log_slot=config.LOG_SLOT):
        utils.xprint('Training ...', newline=True, logger=self.logger)
        timer = utils.Timer()

        loss_epoch = numpy.zeros((0,))
        deviation_epoch = numpy.zeros((0,))
        params = None
        str_params = None
        # for iepoch in range(self.num_epoch):
        iepoch = 0
        while True:

            utils.xprint('  Epoch %d ...' % iepoch, newline=True, logger=self.logger)
            loss = None
            deviations = None
            loss_batch = numpy.zeros((0,))
            deviation_batch = numpy.zeros((0,))
            ibatch = 0
            while True:
                try:

                    # retrieve 1 batch for each node
                    instants, inputs, targets = self.sampler.load_batch(True)
                    if inputs is None:
                        self.sampler.reset_entry()
                        break

                    def format_netflow():
                        embedded = self.check_embedded(inputs)
                        hids = []
                        for ihid in xrange(self.dimension_hidden_layers[0]):
                            check_hid = self.checks_hid[ihid]
                            hids += [check_hid(inputs)]
                        netout = self.check_outputs(inputs)

                        string = ''
                        string += utils.format_var(embedded[0, 0], name='embedded[0][0]') + '\n'
                        for ihid in xrange(self.dimension_hidden_layers[0]):
                            string += utils.format_var(hids[ihid][0][0], name='hidden-%d[0][0]' % ihid) + '\n'
                        string += utils.format_var(netout[0, 0], 'netout[0][0]') + '\n'
                        return string

                    utils.xprint('    Batch %d ...' % ibatch, level=2, logger=self.logger)

                    # todo log parameter values to file
                    if params is None:
                        params = self.check_params()
                        str_params = utils.format_var(params, self.param_names)
                    str_netflow = format_netflow()
                    self.network_history[ibatch] = {'params': str_params, 'netflow': str_netflow}

                    probs = self.check_probs(inputs, targets)
                    try:
                        utils.assert_finite(probs, 'probs')
                    except Exception, e:
                        e.message += '\n'
                        e.message += '%s' % str_netflow
                        raise
                    else:
                        pass


                    # prediction = self.func_predict(inputs)
                    deviations = self.func_compare(inputs, targets)

                    loss = self.func_train(inputs, targets)
                    params = self.check_params()
                    str_params = utils.format_var(params, self.param_names)
                    try:
                        utils.assert_finite(params, 'params')
                    except Exception, e:
                        info = """
                                    Unvalid training ... %s
                                    Before this training:
                                    %s
                                    After:
                                    %s
                                """ % (utils.format_var(float(loss), name='loss'),
                                       self.network_history[ibatch]['params'],
                                       str_params)
                        e.message += info
                        raise
                    else:
                        pass

                    loss_batch = numpy.append(loss_batch, loss)
                    deviation_batch = numpy.append(deviation_batch, numpy.mean(deviations))

                    self.logger.log({'epoch': iepoch, 'batch': ibatch,
                                     'loss': utils.format_var(float(loss)), 'deviations': utils.format_var(deviations)}, name="training")

                    if divmod(ibatch, log_slot)[1] == 0:
                        # utils.xprint('    Batch %d ...' % ibatch, level=1, logger=self.logger)
                        utils.xprint(utils.format_var(float(loss), name='loss') + '; ' + utils.format_var(deviations, name='deviations'),
                                     level=1, newline=True, logger=self.logger)

                    ibatch += 1

                except KeyboardInterrupt, e:
                    raise

            if divmod(ibatch, log_slot)[1] != 0:
                utils.xprint('    Batch %d ...' % ibatch, level=1, logger=self.logger)
                utils.xprint(utils.format_var(float(loss), name='loss') + '; ' + utils.format_var(deviations, name='deviations'),
                             level=1, newline=True, logger=self.logger)

            loss_epoch = numpy.append(loss_epoch, loss_batch[-1])
            deviation_epoch = numpy.append(deviation_epoch, deviation_batch[-1])

            iepoch += 1
            if iepoch >= self.num_epoch:
                more = utils.confirm("Try more epochs?")
                if more:
                    num_more = utils.ask_int("How many?")
                    if num_more is not None and num_more > 0:
                        self.num_epoch += num_more
                    else:
                        break
                else:
                    break

        utils.xprint('Done in %s' % timer.stop(), newline=True, logger=self.logger)
        return loss_epoch

    def export_params(self, path=None):
        try:
            FILENAME_EXPORT = 'params.pkl'

            timer = utils.Timer()
            if self.params_all is None:
                raise RuntimeError("save_params @ SharedLSTM: Cannot save parameters before the model is built.")

            if path is None:
                path = filer.format_path(self.logger.log_path, FILENAME_EXPORT)

            utils.xprint("Exporting parameters to '%s' ... " % path, logger=self.logger)

            if self.check_params is None:
                self.check_params = theano.function([], self.params_all, allow_input_downcast=True)

            params_all = self.check_params()

            filer.dump_to_file(path, params_all)

            utils.xprint('Done in %s' % timer.stop(), newline=True, logger=self.logger)
            return path

        except:
            raise

    def import_params(self, path=None):
        try:
            if self.params_all is None:
                raise RuntimeError("save_params @ SharedLSTM: Cannot export parameters before the model is built.")

            if path is None:
                path = utils.ask_path('Import from which file?', assert_exist=True)
                if path is None:
                    return

            params_all = filer.load_from_file(path)
            self.set_params(params_all)

        except:
            raise

    def set_params(self, params):
        try:
            timer = utils.Timer()
            utils.xprint('Importing given parameters ...', logger=self.logger)
            L.layers.set_all_param_values(self.network, params)
            utils.xprint('Done in %s' % timer.stop(), newline=True, logger=self.logger)

        except:
            raise

    def complete(self):
        self.logger.complete()

    @staticmethod
    def test():

        timestamp = utils.get_timestamp()
        sub_logger = filer.Logger(identifier=timestamp)
        sub_logger.copy_config()
        sub_logger.register_console()

        try:

            def test_bivar_norm():
                _prob = SharedLSTM._check_bivar_norm([2000, -2000], [100, -100, 10000, 15000, 0])
                _prob = SharedLSTM._check_bivar_norm([2000, -2000], [100, -100, 10000, 15000, 0.1])
                _prob = SharedLSTM._check_bivar_norm([2000, -2000], [100, -100, 10000, 15000, -0.1])
                _prob = SharedLSTM._check_bivar_norm([2000, -2000], [100, -100, 10000, 15000, 1.e-8])
                _prob = SharedLSTM._check_bivar_norm([2000, -2000], [100, -100, 10000, 15000, 0.9])
                _prob = SharedLSTM._check_bivar_norm([2000, -2000], [100, -100, 10000, 15000, -0.9])

            if __debug__:
                theano.config.exception_verbosity = 'high'
                theano.config.optimizer = 'fast_compile'

            sampler = Sampler(nodes=3, keep_positive=True)
            half = Sampler.clip(sampler, indices=(sampler.length / 2))
            model = SharedLSTM(sampler=half, motion_range=sampler.motion_range, logger=sub_logger)

            outputs_var, params_var = model.build_network()
            predict, compare, train = model.compile()

            check_e, checks_hid, check_outputs, check_params, check_probs = model.get_checks()

            root_logger = filer.Logger()
            root_logger.register("loss", tags=["timestamp", "loss-by-epoch"])

            loss = model.train()
            path_params = model.export_params()

            def test_importing():
                model2 = SharedLSTM(sampler=half, motion_range=sampler.motion_range, logger=sub_logger)
                params = filer.load_from_file(path_params)
                model2.build_network(params=params)
                model2.compile()

                model3 = SharedLSTM(sampler=half, motion_range=sampler.motion_range, logger=sub_logger)
                model3.build_network()
                model3.import_params()
                model3.compile()

            # test_importing()

            root_logger.log({"timestamp": timestamp, "loss-by-epoch": '%s\n' % utils.format_var(loss, detail=True)},
                            name="loss")
            model.complete()

        except Exception, e:
            utils.handle(e, logger=sub_logger)
            raise
