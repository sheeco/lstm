# coding:GBK

import numpy
import scalar
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


def social_mask(sequences):
    """
    Calculate social hidden-state tensor H for single batch of training sequences for all nodes
    :param sequences: [N_NODES, SIZE_BATCH, LENGTH_SEQUENCE_INPUT, 2]
    :return: [N_NODES, SIZE_BATCH, LENGTH_SEQUENCE_INPUT, RANGE_NEIGHBORHOOD, RANGE_NEIGHBORHOOD, N_NODES]
    """

    # mn_pool = numpy.zeros((1, 2), dtype=int)
    # distance_xy = numpy.zeros((1, 2), dtype=float)
    # indication = lambda mn_pool, distance_xy: sum((distance_xy >= SIZE_POOL * (mn_pool - RANGE_NEIGHBORHOOD / 2))
    #                      & (distance_xy < SIZE_POOL * (mn_pool - RANGE_NEIGHBORHOOD / 2 + 1))) == 2

    # mn_pool = T.bvector('mn')
    # distance_xy = T.fvector('distance')
    # # todo test
    # # 1 / 0
    # if_within_grid = T.eq(T.sum((distance_xy >= SIZE_POOL * (mn_pool - RANGE_NEIGHBORHOOD / 2))
    #                      & (distance_xy < SIZE_POOL * (mn_pool - RANGE_NEIGHBORHOOD / 2 + 1))), 2)
    # # distance_xy = other_xy - my_xy
    # # pass in [m, n] in [0, RANGE_NEIGHBORHOOD)
    # indication = theano.function([mn_pool, distance_xy], if_within_grid, allow_input_downcast=True)

    # todo test
    # distance_xy = other_xy - my_xy
    # pass in [m, n] in [0, RANGE_NEIGHBORHOOD)
    # 1 / 0
    indication = lambda mn_pool, distance_xy: (T.ge(distance_xy[0], SIZE_POOL * (mn_pool[0] - RANGE_NEIGHBORHOOD / 2)))\
                                              and T.lt(distance_xy[0], SIZE_POOL * (mn_pool[0] - RANGE_NEIGHBORHOOD / 2 + 1))\
                                              and (T.ge(distance_xy[1], SIZE_POOL * (mn_pool[1] - RANGE_NEIGHBORHOOD / 2)))\
                                              and T.lt(distance_xy[1], SIZE_POOL * (mn_pool[1] - RANGE_NEIGHBORHOOD / 2 + 1))

    # n_nodes = all_nodes.shape[0]
    # n_nodes = T.cast(n_nodes, 'int32')
    # n_nodes = n_nodes.eval()

    ret = T.zeros((N_NODES, SIZE_BATCH, LENGTH_SEQUENCE_INPUT, RANGE_NEIGHBORHOOD, RANGE_NEIGHBORHOOD, N_NODES))
    for inode in xrange(N_NODES):
        for ibatch in xrange(SIZE_BATCH):
            for iseq in xrange(LENGTH_SEQUENCE_INPUT):
                for m in xrange(RANGE_NEIGHBORHOOD):
                    for n in xrange(RANGE_NEIGHBORHOOD):
                        for jnode in xrange(N_NODES):
                            if jnode == inode:
                                continue
                            ind = indication([m, n], sequences[jnode, ibatch, iseq] - sequences[inode, ibatch, iseq])
                            T.set_subtensor(ret[inode, ibatch, iseq, m, n, jnode], ind)

    return ret


def build_shared_lstm(input_var=None):

    try:

        print 'Building shared LSTM network ...',

        # [(x, y)]
        layer_in = L.layers.InputLayer(name="input-layer", input_var=input_var,
                                       shape=(N_NODES, SIZE_BATCH, LENGTH_SEQUENCE_INPUT, DIMENSION_SAMPLE))
        # e = relu(x, y; We)
        layer_e = L.layers.DenseLayer(layer_in, name="e-layer", num_units=DIMENSION_EMBED_LAYER,
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

        layer_distribution = L.layers.DenseLayer(layer_h, name="distribution-layer", num_units=5,
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


# def test_model():

    # all_samples, all_targets = load_batch_for_nodes(read_traces_from_path(PATH_TRACE_FILES),
    #                                                        SIZE_BATCH, [], 0, True)

    # l_social_pooling = InputLayer(shape=(SIZE_BATCH, LENGTH_SEQUENCE_INPUT, DIMENSION_SAMPLE))
    # First, we build the network, starting with an input layer
    # Recurrent layers expect input of shape
    # (batch_size, SEQ_LENGTH, num_features)

    # We now build the LSTM layer which takes l_in as the input layer
    # We clip the gradients at GRAD_CLIP to prevent the problem of exploding gradients.

    # # only_return_final = True?
    # l_forward_1 = LSTMLayer(
    #     layer_in, DIMENSION_HIDDEN_LAYERS, grad_clipping=GRAD_CLIP,
    #     nonlinearity=tanh)
    #
    # # Parameter sharing between multiple layers can be achieved by using the same Theano shared variable instance
    # # for their parameters. e.g.
    # #
    # # l1 = DenseLayer(l_in, num_units=100)
    # # l2 = DenseLayer(l_in, num_units=100, W=l1.W)
    #
    # # DenseLayer: full connected
    # # The output of l_forward_2 of shape (batch_size, N_HIDDEN) is then passed through the dense layer to create
    # # The output of this stage is (size_batch, dim_sample)
    # l_out = DenseLayer(l_forward_2, num_units=1, W=Normal())

    # print("Building network ...")

    # # Theano tensor for the targets
    # target_values = T.fmatrix('target_output')
    #
    # # network_output: [size_batch, dim_sample]
    # # get_output produces a variable for the output of the net
    # # network_output = get_output(l_out)
    #
    # # whats categorical cross-entropy?
    # # The loss function is calculated as the mean of the (categorical) cross-entropy between the prediction and target.
    # cost = T.nnet.categorical_crossentropy(network_output, target_values).mean()
    #
    # # Retrieve all parameters from the network
    # all_params = get_all_params(temp_layer_out, trainable=True)
    #
    # # Compute RMSProp updates for training
    # print("Computing updates ...")
    # updates = L.updates.rmsprop(cost, all_params, LEARNING_RATE_RMSPROP)
    #
    # # Theano functions for training and computing cost
    # print("Compiling functions ...")
    # train = theano.function([layer_in.input_var, target_values], cost, updates=updates, allow_input_downcast=True)
    # compute_cost = theano.function([layer_in.input_var, target_values], cost, allow_input_downcast=True)
    #
    # predict = theano.function([layer_in.input_var], network_output, allow_input_downcast=True)
    #
    # def try_it_out():
    #     preds = numpy.zeros((node_count, DIMENSION_SAMPLE))
    #     ins, tars = load_batch_for_nodes(all_traces, 1, [], 0, True)
    #
    #     for i in range(LENGTH_SEQUENCE_OUTPUT):
    #         for inode in range(node_count):
    #             preds[inode] = predict(ins[inode])
    #             print preds[inode], tars[inode, :, LENGTH_SEQUENCE_OUTPUT - 1, :]
    #
    # print("Training ...")
    # p = 0
    # try:
    #     for it in xrange(N_SEQUENCES * NUM_EPOCH / SIZE_BATCH):
    #         try_it_out()  # Generate text using the p^th character as the start.
    #
    #         avg_cost = 0
    #         for _ in range(LOG_SLOT):
    #             for node in range(N_NODES):
    #                 # 获取(输入序列,实际输出)配对
    #                 inputs, targets = load_batch_for_nodes(all_traces, SIZE_BATCH, [], p, True)
    #
    #                 p += LENGTH_SEQUENCE_INPUT + SIZE_BATCH - 1
    #                 if p + SIZE_BATCH + LENGTH_SEQUENCE_INPUT >= N_SEQUENCES:
    #                     print('Carriage Return')
    #                     p = 0
    #
    #                 # 训练
    #                 avg_cost += train(inputs[node], targets[node, :, LENGTH_SEQUENCE_OUTPUT - 1, :])
    #             print("Epoch {} average loss = {}".format(it * 1.0 * LOG_SLOT / N_SEQUENCES * SIZE_BATCH,
    #                                                       avg_cost / LOG_SLOT))
    #     try_it_out()

    # except KeyboardInterrupt:
    #     pass


def compute_and_compile(network, inputs_in, targets_in):

    try:

        print 'Preparing ...',

        network_outputs = L.layers.get_output(network)

        # Use mean(x, y) as predictions directly
        predictions = network_outputs[:, :, :, 0:2]
        # Remove time column
        facts = targets_in[:, :, :, 1:3]

        """Euclidean Error for Observation"""

        # Elemwise differences
        differences = T.sub(predictions, facts)
        original_shape = differences.shape
        new_shape = (original_shape[0] * original_shape[1] * original_shape[2], original_shape[3])
        differences = T.reshape(differences, new_shape)
        error = T.add(differences[:, 0] ** 2, differences[:, 1] ** 2) ** 0.5
        new_shape = (original_shape[0] * original_shape[1] * original_shape[2], 1)
        error = T.reshape(differences, new_shape)

        """NNL Loss for Training"""

        # Reshape for convenience
        original_shape = facts.shape
        new_shape = (original_shape[0] * original_shape[1] * original_shape[2], original_shape[3])
        targets = T.reshape(facts, new_shape)
        original_shape = network_outputs.shape
        new_shape = (original_shape[0] * original_shape[1] * original_shape[2], original_shape[3])
        distributions = T.reshape(network_outputs, new_shape)
        # distributions = T.constant([[50, 100, 0.04, 0.09, 1], [50, 100, 0.18, 0.08, 0.5]])

        # Use scan to replace loop with tensors
        def step_loss(idx, distribution_mat, target_mat):

            # From the idx of the start of the slice, the vector and the length of
            # the slice, obtain the desired slice.
            distribution = distribution_mat[idx:idx + 1, :]
            target = target_mat[idx:idx + 1, :]

            mean = distribution[:, 0:2]
            deviation = distribution[0:1, 2:4]
            correlation = distribution[0:1, 4]

            # deviation_var = T.dvector('deviation')
            # correlation_var = T.dvector('correlation')
            # covariance_var = T.dot(T.transpose(deviation_var) ** 0.5,
            #                        T.mul(T.extra_ops.repeat(correlation_var, 2, axis=0), deviation_var) ** 0.5)
            # compute_covariance = theano.function([deviation_var, correlation_var], covariance_var)

            from breze.arch.component.distributions.mvn import logpdf

            # sample_var = T.dmatrix('sample')
            # mean_var = T.dvector('mean')
            # cov_var = T.dmatrix('cov')
            # # p = pdf(msample, vmean, mcov)
            # # pdf_multi_norm = theano.function([msample, vmean, mcov], p)
            # logp_var = logpdf(sample_var, mean_var, cov_var)
            # logpdf_multi_norm = theano.function([sample_var, mean_var, cov_var], logp_var)

            covariance = T.dot(T.transpose(deviation) ** 0.5,
                                   T.mul(T.extra_ops.repeat(correlation, 2, axis=0), deviation) ** 0.5)
            # Normal Negative Log-likelihood
            nnl = logpdf(target, mean, covariance)

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
        compare = theano.function([inputs_in, targets_in], error, allow_input_downcast=True)
        train = theano.function([inputs_in, targets_in], loss, updates=updates, allow_input_downcast=True)

        print 'Done'
        return predict, compare, train

    except (KeyboardInterrupt, SystemExit):
        pass


if __name__ == '__main__':

    # todo build the network
    # todo define initializers
    # todo add debug info & assertion
    # todo extract args into config
    # todo write config to .log
    # todo read config from command line args

    input_var = T.tensor4("input", dtype='float64')
    # target_var = T.fvector("targets")
    target_var = T.tensor4("target", dtype='float64')
    # net_targets = T.constant(targets, "net targets")
    network = build_shared_lstm(input_var)

    predict, loss, train = compute_and_compile(network, input_var, target_var)

    for iepoch in range(NUM_EPOCH):
        p_entry = 0
        while True:
            # 1 batch for each node
            instants, inputs, targets = load_batch_for_nodes(all_traces, SIZE_BATCH, N_NODES, p_entry, True)
            if inputs is None:
                break

            p_entry += SIZE_BATCH
            train(inputs, targets)

    print 'Exit'