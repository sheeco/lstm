# coding:GBK

import numpy
import theano
import theano.tensor as T
import lasagne as L

from config import *
from utils import *
from sample import *


all_traces = read_traces_from_path(PATH_TRACE_FILES)
MAX_SEQUENCES = len(all_traces.items())

N_NODES = len(all_traces)
# if N_NODES_EXPECTED > N_NODES:
#     raise RuntimeError("Cannot find enough nodes in the given path.")
N_NODES = N_NODES_EXPECTED if N_NODES_EXPECTED < N_NODES else N_NODES

RMSPROP = lambda loss, params: L.updates.rmsprop(loss, params, LEARNING_RATE_RMSPROP)
METRIC_TRAINING = RMSPROP

CROSS_ENTROPY = lambda pred, target: L.objectives.categorical_crossentropy(pred, target)
SQUARED = lambda pred, target: L.objectives.squared_error(pred, target)
METRIC_LOSS = SQUARED


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
        layer_in = L.layers.InputLayer(name="input-in", input_var=input_var,
                                             shape=(N_NODES, SIZE_BATCH, LENGTH_SEQUENCE_INPUT, DIMENSION_SAMPLE))
        # e = relu(x, y; We)
        layer_e = L.layers.DenseLayer(layer_in, name="e", num_units=DIMENSION_EMBED_LAYER,
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
        # not sure about whether to reshape or not
        layer_h = L.layers.ReshapeLayer(layer_concated_lstms, (N_NODES, -1, [1], [2]))
        assert match(layer_h.output_shape, (N_NODES, SIZE_BATCH, LENGTH_SEQUENCE_INPUT, DIMENSION_HIDDEN_LAYERS))

        # simple decoder
        layer_decoded = L.layers.DenseLayer(layer_h, name="e", num_units=DIMENSION_SAMPLE,
                                            nonlinearity=L.nonlinearities.rectify, num_leading_axes=3)
        assert match(layer_decoded.output_shape,
                     (N_NODES, SIZE_BATCH, LENGTH_SEQUENCE_INPUT, DIMENSION_SAMPLE))

        layer_output = L.layers.SliceLayer(layer_decoded, slice(-1, None), axis=-2)
        assert match(layer_output.output_shape,
                     (N_NODES, SIZE_BATCH, 1, DIMENSION_SAMPLE))

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


def compute_and_compile(network, input_var, target_var):

    try:

        print 'Computing updates ...',

        predictions = L.layers.get_output(network)

        # The loss function is calculated as the mean of the (categorical) cross-entropy between the prediction and target.
        loss = METRIC_LOSS(predictions, target_var)
        loss = loss.mean()

        # Retrieve all parameters from the network
        params = L.layers.get_all_params(network, trainable=True)

        # Compute RMSProp updates for training
        updates = METRIC_TRAINING(loss, params)

        print 'Done'
        print 'Compiling functions ...',

        # Theano functions for training and computing cost
        predict = theano.function([input_var], predictions, allow_input_downcast=True)
        compute_loss = theano.function([input_var, target_var], loss, allow_input_downcast=True)
        # train = theano.function([input_var], loss, givens={net_targets: targets}, updates=updates, allow_input_downcast=True)
        train = theano.function([input_var, target_var], loss, updates=updates, allow_input_downcast=True)

        print 'Done'
        return predict, compute_loss, train

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
            if len(inputs) == 0:
                break

            p_entry += SIZE_BATCH
            train(inputs, targets)

    print 'Exit'