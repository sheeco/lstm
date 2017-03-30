# coding:GBK

import numpy
import theano
import theano.tensor as T
import lasagne as L

import config
import utils
from Sampler import *

__all__ = [
    'SharedLSTM'
    ]

# todo add timer
# todo add debug info & assertion
# todo write config to .log
# todo read config from command line args


class SharedLSTM:

    def __init__(self, sampler=None, inputs=None, targets=None, dimension_embed_layer=config.DIMENSION_EMBED_LAYER,
                 dimension_hidden_layers=config.DIMENSION_HIDDEN_LAYERS, grad_clip=config.GRAD_CLIP, num_epoch=config.NUM_EPOCH,
                 learning_rate_rmsprop=config.LEARNING_RATE_RMSPROP, rho_rmsprop=config.RHO_RMSPROP,
                 epsilon_rmsprop=config.EPSILON_RMSPROP, check=__debug__):
        try:
            if sampler is None:
                sampler = Sampler()
                sampler.pan_to_positive()
            self.sampler = sampler
            self.sampler.reset_entry()
            self.num_node = self.sampler.num_node

            self.dimension_sample = self.sampler.dimension_sample
            self.length_sequence_input = self.sampler.length_sequence_input
            self.length_sequence_output = self.sampler.length_sequence_output
            self.size_batch = self.sampler.size_batch
            self.dimension_embed_layer = dimension_embed_layer
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

            self.layer_in = None
            self.layer_e = None
            self.layer_h = None
            self.layer_out = None

            self.outputs = None
            self.params = None
            self.updates = None
            self.predictions = None
            self.probabilities = None
            self.loss = None
            self.deviations = None

            self.func_predict = None
            self.func_compare = None
            self.func_train = None

            self.check = check
            self.check_e = None
            self.check_netout = None
            self.check_params = None
            self.check_probs = None

            self.w_e = L.init.Uniform(std=0.005, mean=(1. / self.dimension_sample))
            # self.w_e = L.init.Uniform(range=(0., 1.))
            self.b_e = L.init.Constant(0.)
            self.f_e = None

            self.w_lstm_in = L.init.Uniform(std=0.005, mean=(1. / self.dimension_sample))
            self.w_lstm_hid = L.init.Uniform(std=0.005, mean=(1. / self.dimension_hidden_layers))
            self.w_lstm_cell = L.init.Uniform(std=0.005, mean=(1. / self.dimension_hidden_layers))
            self.b_lstm = L.init.Constant(0.)
            self.f_lstm_hid = L.nonlinearities.softplus
            self.f_lstm_cell = L.nonlinearities.softplus
            self.init_lstm_hid = L.init.Constant(0.)
            self.init_lstm_cell = L.init.Constant(0.)

            self.w_means = L.init.Uniform(std=0.005, mean=(1. / self.dimension_hidden_layers))
            self.b_means = L.init.Constant(0.)
            self.f_means = None
            self.w_deviations = L.init.Uniform(std=0.1, mean=(100. / self.dimension_hidden_layers / self.num_node))
            self.b_deviations = L.init.Constant(0.)
            self.f_deviations = L.nonlinearities.softplus
            # self.w_correlation = L.init.Uniform(std=0.0005, mean=0.)
            self.w_correlation = L.init.Uniform(std=0., mean=0.)
            self.b_correlation = L.init.Constant(0.)
            # self.f_correlation = L.nonlinearities.tanh
            self.f_correlation = SharedLSTM.scaled_tanh

        except:
            raise

    @staticmethod
    def bivar_norm(x1, x2, mu1, mu2, sigma1, sigma2, rho):
        # pdf of bivariate norm
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
        # ([x1, x2], [mu1, mu2, sigma1, sigma2, rho])
        try:
            _prob = SharedLSTM.bivar_norm(fact[0], fact[1], distribution[0], distribution[1], distribution[2], distribution[3],
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

    # todo change to pure theano
    # todo add multi-layer
    def build_network(self):
        try:

            print 'Building shared LSTM network ...',

            # [(x, y)]
            layer_in = L.layers.InputLayer(name="input-layer", input_var=self.inputs,
                                           shape=(self.num_node, self.size_batch, self.length_sequence_input, self.dimension_sample))
            # e = relu(x, y; We)
            layer_e = L.layers.DenseLayer(layer_in, name="e-layer", num_units=self.dimension_embed_layer, W=self.w_e, b=self.b_e,
                                          nonlinearity=self.f_e, num_leading_axes=3)
            assert utils.match(layer_e.output_shape, (self.num_node, self.size_batch, self.length_sequence_input, self.dimension_embed_layer))

            layers_in_lstms = []
            for inode in xrange(0, self.num_node):
                layers_in_lstms += [L.layers.SliceLayer(layer_e, inode, axis=0)]
            assert all(
                utils.match(ilayer_in_lstm.output_shape, (self.size_batch, self.length_sequence_input, self.dimension_embed_layer)) for
                ilayer_in_lstm
                in layers_in_lstms)

            # Create an LSTM layers for the 1st node
            layer_lstm_0 = L.layers.LSTMLayer(layers_in_lstms[0], self.dimension_hidden_layers, name="LSTM-0",
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
            assert utils.match(layer_lstm_0.output_shape, (self.size_batch, self.length_sequence_input, self.dimension_hidden_layers))

            layers_lstm = [layer_lstm_0]

            # Create params sharing LSTMs for the rest (n - 1) nodes,
            # which have params exactly the same as LSTM_0
            for inode in xrange(1, self.num_node):
                # Overdated implement for lasagne/lasagne
                layers_lstm += [
                    L.layers.LSTMLayer(layers_in_lstms[inode], self.dimension_hidden_layers, name="LSTM-" + str(inode),
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

            layer_concated_lstms = L.layers.ConcatLayer(layers_lstm, axis=0)
            assert utils.match(layer_concated_lstms.output_shape,
                               (self.num_node * self.size_batch, self.length_sequence_input, self.dimension_hidden_layers))

            layer_h = L.layers.ReshapeLayer(layer_concated_lstms, (self.num_node, -1, [1], [2]))
            assert utils.match(layer_h.output_shape, (self.num_node, self.size_batch, self.length_sequence_input, self.dimension_hidden_layers))

            # # simple decoder
            #
            # layer_decoded = L.layers.DenseLayer(layer_h, name="decoded", num_units=self.dimension_sample,
            #                                     nonlinearity=L.nonlinearities.rectify, num_leading_axes=3)
            # assert match(layer_decoded.output_shape,
            #              (N_NODES, self.size_batch, self.length_sequence_input, self.dimension_sample))
            #
            # layer_output = L.layers.SliceLayer(layer_distribution, slice(-1, None), axis=-2)
            # assert match(layer_output.output_shape,
            #              (N_NODES, self.size_batch, 1, self.dimension_sample))

            # layer_distribution = L.layers.DenseLayer(layer_h, name="distribution-layer", num_units=5, W=L.init.GlorotUniform(gain='relu'),
            #                                          nonlinearity=None, num_leading_axes=3)
            layer_means = L.layers.DenseLayer(layer_h, name="means-layer", num_units=2, W=self.w_means, b=self.b_means,
                                              nonlinearity=self.f_means, num_leading_axes=3)
            layer_deviations = L.layers.DenseLayer(layer_h, name="deviations-layer", num_units=2, W=self.w_deviations,
                                                   b=self.b_deviations,
                                                   nonlinearity=self.f_deviations, num_leading_axes=3)
            layer_correlation = L.layers.DenseLayer(layer_h, name="correlation-layer", num_units=1, W=self.w_correlation,
                                                    b=self.b_correlation,
                                                    nonlinearity=self.f_correlation, num_leading_axes=3)
            layer_distribution = L.layers.ConcatLayer([layer_means, layer_deviations, layer_correlation], axis=-1)

            assert utils.match(layer_distribution.output_shape,
                               (self.num_node, self.size_batch, self.length_sequence_input, 5))

            layer_out = L.layers.SliceLayer(layer_distribution, indices=slice(-self.length_sequence_output, None), axis=2)
            assert utils.match(layer_distribution.output_shape,
                               (self.num_node, self.size_batch, self.length_sequence_output, 5))
            # layer_output = L.layers.ReshapeLayer(layer_distribution, (-1, [3]))
            # assert match(layer_distribution.output_shape,
            #              (N_NODES * self.size_batch * self.length_sequence_input, 5))

            # layer_output = L.layers.ExpressionLayer(layer_distribution, binary_gaussian_distribution)
            # assert match(layer_distribution.output_shape,
            #              (N_NODES, self.size_batch, self.length_sequence_input, 2))

            print 'Done'
            self.layer_in = layer_in
            self.layer_e = layer_e
            self.layer_h = layer_h
            self.layer_out = layer_out

            return self.layer_out

        except:
            raise

    # todo extract decode() from compile()
    def compile(self):

        try:

            print 'Preparing ...',

            outputs = L.layers.get_output(self.layer_out)

            # Use mean(x, y) as predictions directly
            predictions = outputs[:, :, :, 0:2]
            # Remove time column
            facts = self.targets[:, :, :, 1:3]
            shape_facts = facts.shape
            shape_stacked_facts = (shape_facts[0] * shape_facts[1] * shape_facts[2], shape_facts[3])

            """Euclidean Error for Observation"""

            # Elemwise differences
            differences = T.sub(predictions, facts)
            differences = T.reshape(differences, shape_stacked_facts)
            deviations = T.add(differences[:, 0] ** 2, differences[:, 1] ** 2) ** 0.5
            shape_deviations = (shape_facts[0], shape_facts[1], shape_facts[2])
            deviations = T.reshape(deviations, shape_deviations)

            """NNL Loss for Training"""

            # Reshape for convenience
            facts = T.reshape(facts, shape_stacked_facts)
            shape_distributions = outputs.shape
            shape_stacked_distributions = (shape_distributions[0] * shape_distributions[1] * shape_distributions[2], shape_distributions[3])
            distributions = T.reshape(outputs, shape_stacked_distributions)

            # todo confine deviations to [0, 1] & then scale to motion range
            # Use scan to replace loop with tensors
            def step_loss(idx, distribution_mat, fact_mat):

                # From the idx of the start of the slice, the vector and the length of
                # the slice, obtain the desired slice.

                distribution = distribution_mat[idx, :]
                target = fact_mat[idx, :]
                prob = SharedLSTM.bivar_norm(target[0], target[1], distribution[0], distribution[1], distribution[2],
                                             distribution[3], distribution[4])

                # Do something with the slice here. I don't know what you want to do
                # to I'll just return the slice itself.

                return prob

            # Make a vector containing the start idx of every slice
            indices = T.arange(facts.shape[0])

            probs, updates_loss = theano.scan(fn=step_loss,
                                              sequences=[indices],
                                              non_sequences=[distributions, facts])

            # # Normal Negative Log-likelihood
            nnls = T.neg(T.log(probs))
            loss = T.sum(nnls)
            # loss = T.mean(deviations)

            print 'Done'
            print 'Computing updates ...',

            # Retrieve all parameters from the self.layer_out
            params = L.layers.get_all_params(self.layer_out, trainable=True)

            # Compute RMSProp updates for training
            RMSPROP = L.updates.rmsprop(loss, params, learning_rate=self.learning_rate_rmsprop, rho=self.rho_rmsprop,
                                        epsilon=self.epsilon_rmsprop)
            updates = RMSPROP

            self.params = params
            self.updates = updates
            self.outputs = outputs
            self.predictions = predictions
            self.probabilities = probs
            self.loss = loss
            self.deviations = deviations

            print 'Done'

            print 'Compiling functions ...',

            # Theano functions for training and computing cost
            self.func_predict = theano.function([self.inputs], self.predictions, allow_input_downcast=True)
            self.func_compare = theano.function([self.inputs, self.targets], self.deviations, allow_input_downcast=True)
            self.func_train = theano.function([self.inputs, self.targets], self.loss, updates=updates, allow_input_downcast=True)

            if self.check:
                e = L.layers.get_output(self.layer_e)
                self.check_e = theano.function([self.inputs], e, allow_input_downcast=True)
                self.check_netout = theano.function([self.inputs], self.outputs, allow_input_downcast=True)
                self.check_params = theano.function([], self.params, allow_input_downcast=True)
                self.check_probs = theano.function([self.inputs, self.targets], self.probabilities, allow_input_downcast=True)

            print 'Done'
            return self.func_predict, self.func_compare, self.func_train

        except:
            raise

    def get_checks(self):

        return self.check_e, self.check_netout, self.check_params, self.check_probs

    def train(self):
        try:
            print 'Training ...'

            loss_epoch = numpy.zeros((self.num_epoch, 3))
            deviation_epoch = numpy.zeros((self.num_epoch, 3))
            for iepoch in range(self.num_epoch):
                print '\tEpoch %d ... ' % iepoch,
                loss_batch = numpy.zeros((0,))
                deviation_batch = numpy.zeros((0,))
                while True:
                    # 1 batch for each node
                    instants, inputs, targets = self.sampler.load_batch(True)
                    if inputs is None:
                        self.sampler.reset_entry()
                        break

                    params = self.check_params()
                    e = self.check_e(inputs)
                    netout = self.check_netout(inputs)
                    probs = self.check_probs(inputs, targets)
                    assert numpy.isfinite(probs).any()

                    # def _check():
                    #     shape = netout.shape
                    #     for i in xrange(shape[0]):
                    #         for j in xrange(shape[1]):
                    #             for k in xrange(shape[2]):
                    #                 _prob = SharedLSTM._check_bivar_norm(targets[i, j, k, 1:3], netout[i, j, k])
                    #
                    # _check()

                    prediction = self.func_predict(inputs)
                    deviation = self.func_compare(inputs, targets)
                    loss = self.func_train(inputs, targets)
                    assert numpy.isfinite(loss).any()

                    loss_batch = numpy.append(loss_batch, loss)
                    deviation_batch = numpy.append(deviation_batch, deviation)

                    # todo check for nan & inf

                loss_epoch[iepoch] = numpy.array(utils.check_range(loss_batch))
                deviation_epoch[iepoch] = numpy.array(utils.check_range(deviation_batch))
                print 'loss: %6s, %6s, %6s' % tuple(['%.0f' % x for x in loss_epoch[iepoch]])
                print '\tdeviation: %6s, %6s, %6s' % tuple(['%.0f' % x for x in deviation_epoch[iepoch]])

        except:
            raise

    @staticmethod
    def test():

        try:

            # _prob = _check_bivar_norm([2000, -2000], [100, -100, 10000, 15000, 0])
            # _prob = _check_bivar_norm([2000, -2000], [100, -100, 10000, 15000, 0.1])
            # _prob = _check_bivar_norm([2000, -2000], [100, -100, 10000, 15000, -0.1])
            # _prob = _check_bivar_norm([2000, -2000], [100, -100, 10000, 15000, 1.e-8])
            # _prob = _check_bivar_norm([2000, -2000], [100, -100, 10000, 15000, 0.9])
            # _prob = _check_bivar_norm([2000, -2000], [100, -100, 10000, 15000, -0.9])

            if __debug__:
                theano.config.exception_verbosity = 'high'

            # sampler = Sampler(path=config.PATH_TRACE_FILES, nodes=3)
            # sampler.pan_to_positive()

            model = SharedLSTM(check=True)
            sampler = model.sampler

            network = model.build_network()
            predict, compare, train = model.compile()

            check_e, check_out, check_params, check_probs = model.get_checks()

            model.train()

        except:
            raise
