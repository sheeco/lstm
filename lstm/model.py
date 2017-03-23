# coding:GBK

import numpy
import theano
import theano.tensor as T
import lasagne as L

from config import *
from utils import *
from sample import *

theano.config.exception_verbosity = 'high'
all_traces = read_traces_from_path(PATH_TRACE_FILES)
MAX_SEQUENCES = len(all_traces.items())

N_NODES = len(all_traces)
# if N_NODES_EXPECTED > N_NODES:
#     raise RuntimeError("Cannot find enough nodes in the given path.")
N_NODES = N_NODES_EXPECTED if N_NODES_EXPECTED < N_NODES else N_NODES


def build_shared_lstm(input_var):

    try:

        print 'Building shared LSTM network ...',

        # [(x, y)]
        layer_in = L.layers.InputLayer(name="input-layer", input_var=input_var,
                                       shape=(N_NODES, SIZE_BATCH, LENGTH_SEQUENCE_INPUT, DIMENSION_SAMPLE))
        # e = relu(x, y; We)
        layer_e = L.layers.DenseLayer(layer_in, name="e-layer", num_units=DIMENSION_EMBED_LAYER, W=L.init.GlorotUniform(), b=L.init.Normal(),
                                      nonlinearity=L.nonlinearities.rectify, num_leading_axes=3)
        assert match(layer_e.output_shape, (N_NODES, SIZE_BATCH, LENGTH_SEQUENCE_INPUT, DIMENSION_EMBED_LAYER))

        layers_in_lstms = []
        for inode in xrange(0, N_NODES):
            layers_in_lstms += [L.layers.SliceLayer(layer_e, inode, axis=0)]
        assert all(
            match(ilayer_in_lstm.output_shape, (SIZE_BATCH, LENGTH_SEQUENCE_INPUT, DIMENSION_EMBED_LAYER)) for ilayer_in_lstm
            in layers_in_lstms)

        # Create an LSTM layers for the 1st node
        layer_lstm_0 = L.layers.LSTMLayer(layers_in_lstms[0], DIMENSION_HIDDEN_LAYERS, name="LSTM-0",
                                          nonlinearity=L.nonlinearities.tanh,
                                          hid_init=L.init.Constant(0.0), cell_init=L.init.Constant(0.0),
                                          only_return_final=False)
        assert match(layer_lstm_0.output_shape, (SIZE_BATCH, LENGTH_SEQUENCE_INPUT, DIMENSION_HIDDEN_LAYERS))

        layers_lstm = [layer_lstm_0]

        # Create params sharing LSTMs for the rest (n - 1) nodes,
        # which have params exactly the same as LSTM_0
        for inode in xrange(1, N_NODES):
            # Overdated implement for lasagne/lasagne
            layers_lstm += [
                L.layers.LSTMLayer(layers_in_lstms[inode], DIMENSION_HIDDEN_LAYERS, name="LSTM-" + str(inode),
                                   grad_clipping=GRAD_CLIP,
                                   nonlinearity=L.nonlinearities.tanh, hid_init=L.init.Constant(0.0),
                                   cell_init=L.init.Constant(0.0), only_return_final=False,
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
                                                      nonlinearity=L.nonlinearities.tanh
                                                      ))]

        layer_concated_lstms = L.layers.ConcatLayer(layers_lstm, axis=0)
        assert match(layer_concated_lstms.output_shape, (N_NODES * SIZE_BATCH, LENGTH_SEQUENCE_INPUT, DIMENSION_HIDDEN_LAYERS))

        layer_h = L.layers.ReshapeLayer(layer_concated_lstms, (N_NODES, -1, [1], [2]))
        assert match(layer_h.output_shape, (N_NODES, SIZE_BATCH, LENGTH_SEQUENCE_INPUT, DIMENSION_HIDDEN_LAYERS))

        # # simple decoder
        #
        # layer_decoded = L.layers.DenseLayer(layer_h, name="decoded", num_units=DIMENSION_SAMPLE,
        #                                     nonlinearity=L.nonlinearities.rectify, num_leading_axes=3)
        # assert match(layer_decoded.output_shape,
        #              (N_NODES, SIZE_BATCH, LENGTH_SEQUENCE_INPUT, DIMENSION_SAMPLE))
        #
        # layer_output = L.layers.SliceLayer(layer_distribution, slice(-1, None), axis=-2)
        # assert match(layer_output.output_shape,
        #              (N_NODES, SIZE_BATCH, 1, DIMENSION_SAMPLE))

        layer_distribution = L.layers.DenseLayer(layer_h, name="distribution-layer", num_units=5, W=L.init.GlorotUniform(), b=L.init.Normal(),
                                                 nonlinearity=None, num_leading_axes=-1)

        assert match(layer_distribution.output_shape,
                     (N_NODES, SIZE_BATCH, LENGTH_SEQUENCE_INPUT, 5))

        layer_output = L.layers.SliceLayer(layer_distribution, indices=slice(-LENGTH_SEQUENCE_OUTPUT, None), axis=2)
        assert match(layer_distribution.output_shape,
                     (N_NODES, SIZE_BATCH, LENGTH_SEQUENCE_OUTPUT, 5))
        # layer_output = L.layers.ReshapeLayer(layer_distribution, (-1, [3]))
        # assert match(layer_distribution.output_shape,
        #              (N_NODES * SIZE_BATCH * LENGTH_SEQUENCE_INPUT, 5))

        # layer_output = L.layers.ExpressionLayer(layer_distribution, binary_gaussian_distribution)
        # assert match(layer_distribution.output_shape,
        #              (N_NODES, SIZE_BATCH, LENGTH_SEQUENCE_INPUT, 2))

        print 'Done'
        return layer_output

    except (KeyboardInterrupt, SystemExit):
        pass

    # except Exception, e:
    #     print str(type(e)) + e.message


def compute_and_compile(network, inputs_in, targets_in):

    try:

        print 'Preparing ...',

        network_outputs = L.layers.get_output(network)

        # Use mean(x, y) as predictions directly
        predictions = network_outputs[:, :, :, 0:2]
        # Remove time column
        facts = targets_in[:, :, :, 1:3]
        shape_facts = facts.shape
        shape_stacked_facts = (shape_facts[0] * shape_facts[1] * shape_facts[2], shape_facts[3])

        """Euclidean Error for Observation"""

        # Elemwise differences
        differences = T.sub(predictions, facts)
        differences = T.reshape(differences, shape_stacked_facts)
        deviations = T.add(differences[:, 0] ** 2, differences[:, 1] ** 2) ** 0.5
        shape_deviations = (shape_facts[0], shape_facts[1], shape_facts[2], 1)
        shape_deviations = (shape_facts[0], shape_facts[1], shape_facts[2])
        deviations = T.reshape(deviations, shape_deviations)

        """NNL Loss for Training"""

        # Reshape for convenience
        targets = T.reshape(facts, shape_stacked_facts)
        shape_distributions = network_outputs.shape
        shape_stacked_distributions = (shape_distributions[0] * shape_distributions[1] * shape_distributions[2], shape_distributions[3])
        distributions = T.reshape(network_outputs, shape_stacked_distributions)
        # distributions = T.constant([[50, 100, 0.04, 0.09, 1], [50, 100, 0.18, 0.08, 0.5]])

        # Use scan to replace loop with tensors
        def step_loss(idx, distribution_mat, target_mat):

            # From the idx of the start of the slice, the vector and the length of
            # the slice, obtain the desired slice.

            # distribution = distribution_mat[idx:idx + 1, :]
            # target = target_mat[idx:idx + 1, :]
            #
            # from breze.arch.component.distributions.mvn import pdf
            #
            # mean = distribution[:, 0:2]
            # deviation = T.abs_(distribution[0:1, 2:4])
            # correlation = distribution[0:1, 4]
            #
            # covariance = T.dot(T.transpose(deviation),
            #                        T.mul(T.extra_ops.repeat(correlation, 2, axis=0), deviation))
            # # Normal Negative Log-likelihood
            # nnl = pdf(target, mean, covariance)

            def bivar_norm(x1, x2, mu1, mu2, sigma1, sigma2, rho):
                # pdf of bivariate norm
                # def pdf(x1, x2):
                # get z
                part1 = (x1 - mu1) ** 2 / sigma1 ** 2
                part2 = - 2. * rho * (x1 - mu1) * (x2 - mu2) / sigma1 * sigma2
                part3 = (x2 - mu2) ** 2 / sigma2 ** 2
                z = part1 + part2 + part3

                cof = 1. / (2. * numpy.pi * sigma1 * sigma2 * T.sqrt(1 - rho ** 2))
                return cof * T.exp(-z / (2. * (1 - rho ** 2)))
                # return pdf

            distribution = distribution_mat[idx, :]
            target = target_mat[idx, :]
            nnl = bivar_norm(target[0], target[1], distribution[0], distribution[1], T.abs_(distribution[2]), T.abs_(distribution[3]), distribution[4])
            # nnl = T.neg(T.log(nnl))

            # Do something with the slice here. I don't know what you want to do
            # to I'll just return the slice itself.

            return nnl

        # Make a vector containing the start idx of every slice
        indices = T.arange(targets.shape[0])

        probs, updates_loss = theano.scan(fn=step_loss,
                                   sequences=[indices],
                                   non_sequences=[distributions, targets])

        loss = T.sum(probs)

        print 'Done'
        print 'Computing updates ...',

        # Retrieve all parameters from the network
        params = L.layers.get_all_params(network, trainable=True)

        # Compute RMSProp updates for training
        RMSPROP = L.updates.rmsprop(loss, params, LEARNING_RATE_RMSPROP)
        updates = RMSPROP

        print 'Done'
        print 'Compiling functions ...',

        # Theano functions for training and computing cost
        predict = theano.function([inputs_in], predictions, allow_input_downcast=True)
        compare = theano.function([inputs_in, targets_in], deviations, allow_input_downcast=True)
        train = theano.function([inputs_in, targets_in], loss, updates=updates, allow_input_downcast=True)
        check = theano.function([inputs_in, targets_in], probs, allow_input_downcast=True)

        print 'Done'
        return predict, compare, train, check

    except (KeyboardInterrupt, SystemExit):
        pass


if __name__ == '__main__':

    try:

        # todo build the network
        # todo define initializers
        # todo add debug info & assertion
        # todo extract args into config
        # todo write config to .log
        # todo read config from command line args

        input_var = T.tensor4("input", dtype='float64')
        target_var = T.tensor4("target", dtype='float64')
        network = build_shared_lstm(input_var)

        predict, compare, train, check = compute_and_compile(network, input_var, target_var)

        for iepoch in range(NUM_EPOCH):
            p_entry = 0
            while True:
                # 1 batch for each node
                instants, inputs, targets = load_batch_for_nodes(all_traces, SIZE_BATCH, N_NODES, p_entry, True)
                if inputs is None:
                    break

                p_entry += SIZE_BATCH
                predictions = predict(inputs)
                deviations = compare(inputs, targets)
                probs = check(inputs, targets)
                loss = train(inputs, targets)

    except (KeyboardInterrupt, SystemExit):
        pass

    print 'Exit'