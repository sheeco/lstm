# coding:GBK
import numpy
import theano
import theano.tensor as T
import lasagne

from config import *
import sample

# class Model(object):
#     def __init__(self):
#         pass

# demo modified based on https://github.com/Lasagne/Recipes/blob/master/examples/lstm_text_generation.py

all_traces = sample.read_traces_from_path(PATH_TRACE_FILES)
data_size = len(all_traces.items())
# node_count = len(all_traces)
node_count = 1


def demo():
    print("Building network ...")

    # First, we build the network, starting with an input layer
    # Recurrent layers expect input of shape
    # (batch_size, SEQ_LENGTH, num_features)

    # (x, y)
    l_in = lasagne.layers.InputLayer(shape=(SIZE_BATCH, LENGTH_SEQUENCE_INPUT, DIMENSION_SAMPLE))
    # e = relu(x, y; We)
    l_e = lasagne.layers.NonlinearityLayer(l_in, nonlinearity=lasagne.nonlinearities.rectify)

    l_social_pooling = lasagne.layers.InputLayer(shape=(SIZE_BATCH, LENGTH_SEQUENCE_INPUT, DIMENSION_SAMPLE))

    all_samples, all_targets = sample.load_batch_for_nodes(sample.read_traces_from_path(PATH_TRACE_FILES),
                                                           SIZE_BATCH, [], 0, True)

    mn_pool = numpy.zeros((1, 2), dtype=int)
    distance_xy = numpy.zeros((1, 2), dtype=float)
    # todo test
    # True / False
    within_grid = sum((distance_xy >= SIZE_POOL * (mn_pool - RANGE_NEIGHBORHOOD / 2)) \
                                    & (distance_xy < SIZE_POOL * (mn_pool - RANGE_NEIGHBORHOOD / 2 + 1))) == 2
    # distance_xy = other_xy - my_xy
    # m, n in [0, RANGE_NEIGHBORHOOD)
    indication = theano.function([mn_pool, distance_xy], within_grid)

    # my_xy = numpy.zeros((1, 2), dtype=float)
    # other_xys = numpy.zeros((N_NODES, 2), dtype=float)
    # matrix_indicator = numpy.zeros((RANGE_NEIGHBORHOOD, RANGE_NEIGHBORHOOD), dtype=int)
    # for inode in xrange(N_NODES):
    #     for m in xrange(RANGE_NEIGHBORHOOD):
    #         for n in xrange(RANGE_NEIGHBORHOOD):
    #             matrix_indicator[m, n] = indication([m, n], other_xys[inode] - my_xy)
    # # todo test
    # indicator = theano.function([my_xy, other_xys], matrix_indicator)

    def social_hidden_tensor(traces, i_my_node, lstm_hiddens):
        """

        :param traces: [N_NODES, SIZE_BATCH, LENGTH_SEQUENCE_INPUT, 2]
        :param i_my_node: int
        :param lstm_hiddens: [N_NODES, SIZE_BATCH, DIMENSION_HIDDEN_LAYERS]
        :return:
        """
        # whats shape of lstm_hiddens?
        ret = numpy.zeros((SIZE_BATCH, LENGTH_SEQUENCE_INPUT, RANGE_NEIGHBORHOOD, RANGE_NEIGHBORHOOD, DIMENSION_HIDDEN_LAYERS))
        my_trace = traces[i_my_node]
        other_traces = numpy.concatenate((traces[:i_my_node], traces[i_my_node + 1:]), 0)
        for ibatch in xrange(SIZE_BATCH):
            for iseq in xrange(LENGTH_SEQUENCE_INPUT):
                for m in xrange(RANGE_NEIGHBORHOOD):
                    for n in xrange(RANGE_NEIGHBORHOOD):
                        for jnode in xrange(traces.shape[0]):
                            if jnode == i_my_node:
                                continue
                            ret[ibatch, iseq, m ,n] += indication([m, n], traces[jnode, ibatch, iseq] - my_trace[jnode, ibatch, iseq]) \
                                                       * lstm_hiddens[jnode, ibatch]
        return ret

    l_a = lasagne.layers.DenseLayer(social_hidden_tensor(all_samples, inode, lstm_hiddens), nonlinearity=lasagne.nonlinearities.rectify)

    l_hidden = lasagne.layers.LSTMLayer([l_e, l_a], DIMENSION_HIDDEN_LAYERS, nonlinearity=lasagne.nonlinearities.tanh)

    # l_in_hid = lasagne.layers.DenseLayer(n_batch, n_steps, n_in = (2, 3, 4)
    # n_hid = 5
    # l_in = lasagne.layers.InputLayer((n_batch, n_steps, n_in))
    # lasagne.layers.InputLayer((None, n_in)), n_hid)
    # l_hid_hid = lasagne.layers.DenseLayer(lasagne.layers.InputLayer((None, n_hid)), n_hid)
    # l_rec = lasagne.layers.CustomRecurrentLayer(l_in, l_in_hid, l_hid_hid)

    # We now build the LSTM layer which takes l_in as the input layer
    # We clip the gradients at GRAD_CLIP to prevent the problem of exploding gradients.

    # only_return_final = True?
    l_forward_1 = lasagne.layers.LSTMLayer(
        l_in, DIMENSION_HIDDEN_LAYERS, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh)

    # Parameter sharing between multiple layers can be achieved by using the same Theano shared variable instance
    # for their parameters. e.g.
    #
    # l1 = lasagne.layers.DenseLayer(l_in, num_units=100)
    # l2 = lasagne.layers.DenseLayer(l_in, num_units=100, W=l1.W)

    # DenseLayer: full connected
    # The output of l_forward_2 of shape (batch_size, N_HIDDEN) is then passed through the dense layer to create
    # The output of this stage is (size_batch, dim_sample)
    l_out = lasagne.layers.DenseLayer(l_forward_2, num_units=1, W=lasagne.init.Normal())

    # Theano tensor for the targets
    target_values = T.fmatrix('target_output')

    # network_output: [size_batch, dim_sample]
    # lasagne.layers.get_output produces a variable for the output of the net
    network_output = lasagne.layers.get_output(l_out)

    # whats categorical cross-entropy?
    # The loss function is calculated as the mean of the (categorical) cross-entropy between the prediction and target.
    cost = T.nnet.categorical_crossentropy(network_output, target_values).mean()

    # Retrieve all parameters from the network
    all_params = lasagne.layers.get_all_params(l_out, trainable=True)

    # Compute RMSProp updates for training
    print("Computing updates ...")
    updates = lasagne.updates.rmsprop(cost, all_params, LEARNING_RATE_RMSPROP)

    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train = theano.function([l_in.input_var, target_values], cost, updates=updates, allow_input_downcast=True)
    compute_cost = theano.function([l_in.input_var, target_values], cost, allow_input_downcast=True)

    predict = theano.function([l_in.input_var], network_output, allow_input_downcast=True)

    def try_it_out():
        preds = numpy.zeros((node_count, DIMENSION_SAMPLE))
        ins, tars = sample.load_batch_for_nodes(all_traces, 1, [], 0, True)

        for i in range(LENGTH_SEQUENCE_OUTPUT):
            for inode in range(node_count):
                preds[inode] = predict(ins[inode])
                print preds[inode], tars[inode, :, LENGTH_SEQUENCE_OUTPUT - 1, :]

    print("Training ...")
    p = 0
    try:
        for it in xrange(data_size * NUM_EPOCH / SIZE_BATCH):
            try_it_out()  # Generate text using the p^th character as the start.

            avg_cost = 0
            for _ in range(LOG_SLOT):
                for node in range(node_count):
                    # 获取(输入序列,实际输出)配对
                    inputs, targets = sample.load_batch_for_nodes(all_traces, SIZE_BATCH, [], p, True)

                    p += LENGTH_SEQUENCE_INPUT + SIZE_BATCH - 1
                    if p + SIZE_BATCH + LENGTH_SEQUENCE_INPUT >= data_size:
                        print('Carriage Return')
                        p = 0

                    # 训练
                    avg_cost += train(inputs[node], targets[node, :, LENGTH_SEQUENCE_OUTPUT - 1, :])
                print("Epoch {} average loss = {}".format(it * 1.0 * LOG_SLOT / data_size * SIZE_BATCH,
                                                          avg_cost / LOG_SLOT))
        try_it_out()

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    demo()
