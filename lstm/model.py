# coding:GBK
import numpy
import theano
import theano.tensor as T
# import lasagne
from lasagne.layers import *
from lasagne.nonlinearities import *
from lasagne.init import *

from config import *
import sample


# class SocialPoolingLayer(MergeLayer):
#     @property
#     def output_shape(self):
#         # input (prev_lstms): [SIZE_BATCH, N_NODES, LENGTH_SEQUENCE_INPUT, DIMENSION_HIDDEN_LAYER]
#         # traces: [SIZE_BATCH, N_NODES, LENGTH_SEQUENCE_INPUT, 2]
#         # output: [SIZE_BATCH, N_NODES, LENGTH_SEQUENCE_INPUT, RANGE_NEIGHBORHOOD, RANGE_NEIGHBORHOOD, DIMENSION_HIDDEN_LAYERS]
#         shape = self.input_shape[0], self.input_shape[2], RANGE_NEIGHBORHOOD, RANGE_NEIGHBORHOOD, self.input_shape[3]
#         if any(isinstance(s, T.Variable) for s in shape):
#             raise ValueError("%s returned a symbolic output shape from its "
#                              "get_output_shape_for() method: %r. This is not "
#                              "allowed; shapes must be tuples of integers for "
#                              "fixed-size dimensions and Nones for variable "
#                              "dimensions." % (self.__class__.__name__, shape))
#         return shape
#
#     def get_output_for(self, input, **kwargs):
#         """
#         Propagates the given input through this layer (and only this layer).
#
#         Parameters
#         ----------
#         input : Theano expression
#             The expression to propagate through this layer.
#
#         Returns
#         -------
#         output : Theano expression
#             The output of this layer given the input to this layer.
#
#
#         Notes
#         -----
#         This is called by the base :meth:`get_output()`
#         to propagate data through a network.
#
#         This method should be overridden when implementing a new
#         :class:`Layer` class. By default it raises `NotImplementedError`.
#         """
#         shape = (None, N_NODES, LENGTH_SEQUENCE_INPUT, DIMENSION_SAMPLE)
#         social_hidden_tensor = numpy.zeros(self.output_shape())
#         for ibatch in xrange(SIZE_BATCH):
#             for inode in xrange(self.input_shape[1]):
#                 social_hidden_tensor[ibatch] = self.social_pool(self.traces, self.traces[inode], self.input_layer)
#         return social_hidden_tensor
#
#     # def get_params(self, unwrap_shared=True, **tags):
#     #     return []
#     # """
#     #
#     # :param incomings: [l_in, [l_prev_lstm]]
#     # :rtype: object
#     # """
#         # def __init__(self, incomings, name=None):
#         #     self.input_shapes = [incoming if isinstance(incoming, tuple)
#         #                          else incoming.output_shape
#         #                          for incoming in incomings]
#         #     self.input_layers = [None if isinstance(incoming, tuple)
#         #                          else incoming
#         #                          for incoming in incomings]
#         #     self.name = name
#         #     self.params = OrderedDict()
#         #     self.get_output_kwargs = []

def match(shape1, shape2):
    return (len(shape1) == len(shape2) and
            all(s1 is None or s2 is None or s1 == s2
                for s1, s2 in zip(shape1, shape2)))


"""
Snippet from https://github.com/Lasagne/Lasagne/issues/584 by @trungnt13
"""
class ExpressionMergeLayer(MergeLayer):

    """
    This layer performs an custom expressions on list of inputs to merge them.
    This layer is different from ElemwiseMergeLayer by not required all
    input_shapes are equal

    Parameters
    ----------
    incomings : a list of :class:`Layer` instances or tuples
        the layers feeding into this layer, or expected input shapes

    merge_function : callable
        the merge function to use. Should take two arguments and return the
        updated value. Some possible merge functions are ``theano.tensor``:
        ``mul``, ``add``, ``maximum`` and ``minimum``.

    output_shape : None, callable, tuple, or 'auto'
        Specifies the output shape of this layer. If a tuple, this fixes the
        output shape for any input shape (the tuple can contain None if some
        dimensions may vary). If a callable, it should return the calculated
        output shape given the input shape. If None, the output shape is
        assumed to be the same as the input shape. If 'auto', an attempt will
        be made to automatically infer the correct output shape.

    Notes
    -----
    if ``output_shape=None``, this layer chooses first input_shape as its
    output_shape

    Example
    --------
    >> from lasagne.layers import InputLayer, DimshuffleLayer, ExpressionMergeLayer
    >> l_in = InputLayer(shape=(None, 500, 120))
    >> l_mask = InputLayer(shape=(None, 500))
    >> l_dim = DimshuffleLayer(l_mask, pattern=(0, 1, 'x'))
    >> l_out = ExpressionMergeLayer(
                                (l_in, l_dim), tensor.mul, output_shape='auto')
    (None, 500, 120)
    """

    def __init__(self, incomings, merge_function, output_shape=None, **kwargs):
        super(ExpressionMergeLayer, self).__init__(incomings, **kwargs)
        if output_shape is None:
            self._output_shape = None
        elif output_shape == 'auto':
            self._output_shape = 'auto'
        elif hasattr(output_shape, '__call__'):
            self.get_output_shape_for = output_shape
        else:
            self._output_shape = tuple(output_shape)

        self.merge_function = merge_function

    def get_output_shape_for(self, input_shapes):
        if self._output_shape is None:
            return input_shapes[0]
        elif self._output_shape is 'auto':
            input_shape = [(0 if s is None else s for s in ishape)
                           for ishape in input_shapes]
            Xs = [T.alloc(0, *ishape) for ishape in input_shape]
            output_shape = self.merge_function(*Xs).shape.eval()
            output_shape = tuple(s if s else None for s in output_shape)
            return output_shape
        else:
            return self._output_shape

    def get_output_for(self, inputs, **kwargs):
        return self.merge_function(*inputs)


def social_mask(all_nodes):
    """
    Calculate social hidden-state tensor H for batch of training samples for all nodes
    :param all_nodes: [SIZE_BATCH, N_NODES, LENGTH_SEQUENCE_INPUT, 2]
    :return: [SIZE_BATCH, N_NODES, LENGTH_SEQUENCE_INPUT, RANGE_NEIGHBORHOOD, RANGE_NEIGHBORHOOD, N_NODES]
    """

    # mn_pool = numpy.zeros((1, 2), dtype=int)
    # distance_xy = numpy.zeros((1, 2), dtype=float)
    # indication = lambda mn_pool, distance_xy: sum((distance_xy >= SIZE_POOL * (mn_pool - RANGE_NEIGHBORHOOD / 2))
    #                      & (distance_xy < SIZE_POOL * (mn_pool - RANGE_NEIGHBORHOOD / 2 + 1))) == 2

    mn_pool = T.ivector('mn')
    distance_xy = T.fvector('distance')
    # todo test
    # 1 / 0
    if_within_grid = T.eq(T.sum((distance_xy >= SIZE_POOL * (mn_pool - RANGE_NEIGHBORHOOD / 2))
                         & (distance_xy < SIZE_POOL * (mn_pool - RANGE_NEIGHBORHOOD / 2 + 1))), 2)
    # distance_xy = other_xy - my_xy
    # pass in [m, n] in [0, RANGE_NEIGHBORHOOD)
    indication = theano.function([mn_pool, distance_xy], if_within_grid, allow_input_downcast=True)

    n_nodes = all_nodes.shape[0]
    ret = numpy.zeros((SIZE_BATCH, n_nodes, LENGTH_SEQUENCE_INPUT, RANGE_NEIGHBORHOOD, RANGE_NEIGHBORHOOD, n_nodes))
    # for ibatch in xrange(SIZE_BATCH):
    for inode in xrange(n_nodes):
        for ibatch in xrange(SIZE_BATCH):
            for iseq in xrange(LENGTH_SEQUENCE_INPUT):
                for m in xrange(RANGE_NEIGHBORHOOD):
                    for n in xrange(RANGE_NEIGHBORHOOD):
                        for jnode in xrange(n_nodes):
                            if jnode == inode:
                                continue
                            ret[ibatch, inode, iseq, m, n, jnode] += indication([m, n],
                                                                                all_nodes[jnode, ibatch, iseq] - all_nodes[inode, ibatch, iseq])

    return ret


def test_model():

    try:

        # todo build the network
        # todo define initializers
        # todo add debug info & assertion
        # todo extract args into config
        # todo write config to .log
        # todo read config from command line args
        all_traces = sample.read_traces_from_path(PATH_TRACE_FILES)
        N_SEQUENCES = len(all_traces.items())

        N_NODES = len(all_traces)
        N_NODES = N_NODES_EXPECTED if N_NODES_EXPECTED < N_NODES else N_NODES
        # 1 batch for each node
        net_inputs, net_targets = sample.load_batch_for_nodes(all_traces, SIZE_BATCH, N_NODES, 0, True)

        print("Building network ...")

        # [(x, y)]
        layer_in = InputLayer(name="input sample",
                                             shape=(N_NODES, None, LENGTH_SEQUENCE_INPUT, DIMENSION_SAMPLE))
        # e = relu(x, y; We)
        layer_e = DenseLayer(layer_in, name="e", num_units=DIMENSION_EMBED_LAYER,
                                            nonlinearity=rectify, num_leading_axes=3)
        assert match(layer_e.output_shape, (N_NODES, None, LENGTH_SEQUENCE_INPUT, DIMENSION_EMBED_LAYER))

        # [N_NODES, SIZE_BATCH, LENGTH_SEQUENCE_INPUT, RANGE_NEIGHBORHOOD, RANGE_NEIGHBORHOOD, N_NODES]
        w_h_to_H = social_mask(net_inputs)
        layer_social_mask = ExpressionLayer(layer_in, social_mask, output_shape=(N_NODES, None, LENGTH_SEQUENCE_INPUT, RANGE_NEIGHBORHOOD, RANGE_NEIGHBORHOOD, N_NODES))
        assert match(layer_social_mask.output_shape,
                     (N_NODES, None, LENGTH_SEQUENCE_INPUT, RANGE_NEIGHBORHOOD, RANGE_NEIGHBORHOOD, N_NODES))

        layer_shuffled_social_mask = DimshuffleLayer(layer_social_mask, (0, 1, 2, 3, 4, 5, 'x'))
        assert match(layer_shuffled_social_mask.output_shape,
                     (N_NODES, None, LENGTH_SEQUENCE_INPUT, RANGE_NEIGHBORHOOD, RANGE_NEIGHBORHOOD, N_NODES, 1))

        layer_prev_h = InputLayer(name="previous h", shape=(N_NODES, None, LENGTH_SEQUENCE_INPUT, DIMENSION_HIDDEN_LAYERS))

        # [N_NODES, SIZE_BATCH, LENGTH_SEQUENCE_INPUT, DIMENSION_HIDDEN_LAYERS]
        # shuffle & broadcast into: [1, SIZE_BATCH, LENGTH_SEQUENCE_INPUT, 1, 1, N_NODES, DIMENSION_HIDDEN_LAYERS] to match social matrix
        layer_shuffled_h = DimshuffleLayer(layer_prev_h, ('x', 1, 2, 'x', 'x', 0, 3))
        assert match(layer_shuffled_h.output_shape,
                     (1, None, LENGTH_SEQUENCE_INPUT, 1, 1, N_NODES, DIMENSION_HIDDEN_LAYERS))

        # todo test lambda
        # Perform elementwise multiplication & sum by -2nd dimension (N_NODES
        # layer_H = ExpressionMergeLayer([layer_shuffled_social_mask, layer_shuffled_h], lambda lt, rt: (T.mul(lt, rt)).sum(-2)
        #                                , output_shape="auto")
        layer_H = ElemwiseMergeLayer([layer_shuffled_social_mask, layer_shuffled_h], T.mul)
        layer_H = ExpressionLayer([layer_H, layer_shuffled_h], lambda x: x.sum(-2), output_shape="auto")

        assert match(layer_H.output_shape, (
        N_NODES, None, LENGTH_SEQUENCE_INPUT, RANGE_NEIGHBORHOOD, RANGE_NEIGHBORHOOD, DIMENSION_HIDDEN_LAYERS))

        # todo reshape batch & node dim together all the time?
        layer_a = DenseLayer(layer_H, name="a", num_units=DIMENSION_EMBED_LAYER,
                                            nonlinearity=rectify, num_leading_axes=3)
        assert match(layer_a.output_shape, (N_NODES, None, LENGTH_SEQUENCE_INPUT, DIMENSION_EMBED_LAYER))
        assert match(layer_e.output_shape, layer_a.output_shape)

        layer_in_lstms = ConcatLayer([layer_e, layer_a], 3, name="e & a")
        assert match(layer_in_lstms.output_shape, (N_NODES, None, LENGTH_SEQUENCE_INPUT, 2 * DIMENSION_EMBED_LAYER))

        layers_in_lstms = []
        for inode in xrange(0, N_NODES):
            layers_in_lstms += [SliceLayer(layer_in_lstms, inode, axis=0)]
        assert all(
            match(ilayer_in_lstm.output_shape, (None, LENGTH_SEQUENCE_INPUT, 2 * DIMENSION_EMBED_LAYER)) for ilayer_in_lstm
            in layers_in_lstms)

        # Create an LSTM layers for the 1st node
        layer_lstm_0 = LSTMLayer(layers_in_lstms[0], DIMENSION_HIDDEN_LAYERS, name="LSTM-0",
                                                nonlinearity=tanh,
                                                hid_init=Constant(0.0), cell_init=Constant(0.0),
                                                only_return_final=False)
        assert match(layer_lstm_0.output_shape, (None, LENGTH_SEQUENCE_INPUT, DIMENSION_HIDDEN_LAYERS))

        layers_lstm = [layer_lstm_0]
        # Create params sharing LSTMs for the rest (n - 1) nodes,
        # which have params exactly the same as LSTM_0
        for inode in xrange(1, N_NODES):
            layers_lstm += [
                LSTMLayer(layers_in_lstms[inode], DIMENSION_HIDDEN_LAYERS, name="LSTM-" + str(inode)
                                         , nonlinearity=tanh, hid_init=Constant(0.0),
                                         cell_init=Constant(0.0), only_return_final=False,
                                         ingate=Gate(W_in=layer_lstm_0.W_in_to_ingate,
                                                                    W_hid=layer_lstm_0.W_hid_to_ingate,
                                                                    W_cell=layer_lstm_0.W_cell_to_ingate,
                                                                    b=layer_lstm_0.b_ingate),
                                         outgate=Gate(W_in=layer_lstm_0.W_in_to_outgate,
                                                                     W_hid=layer_lstm_0.W_hid_to_outgate,
                                                                     W_cell=layer_lstm_0.W_cell_to_outgate,
                                                                     b=layer_lstm_0.b_outgate),
                                         forgetgate=Gate(W_in=layer_lstm_0.W_in_to_forgetgate,
                                                                        W_hid=layer_lstm_0.W_hid_to_forgetgate,
                                                                        W_cell=layer_lstm_0.W_cell_to_forgetgate,
                                                                        b=layer_lstm_0.b_forgetgate),
                                         cell=Gate(W_in=layer_lstm_0.W_in_to_cell,
                                                                  W_hid=layer_lstm_0.W_hid_to_cell,
                                                                  W_cell=None,
                                                                  b=layer_lstm_0.b_cell,
                                                                  nonlinearity=tanh
                                                                  ))]

        # layer_lstms = ListMergeLayer(layers_lstm, name="Merged LSTMs")
        layer_concated_lstms = ConcatLayer(layers_lstm, axis=0)
        assert match(layer_concated_lstms.output_shape, (None, LENGTH_SEQUENCE_INPUT, DIMENSION_HIDDEN_LAYERS))
        layer_h = ReshapeLayer(layer_concated_lstms, (N_NODES, -1, [1], [2]))
        assert match(layer_h.output_shape, (N_NODES, None, LENGTH_SEQUENCE_INPUT, DIMENSION_HIDDEN_LAYERS))

        # layer_h_to_h = NonlinearityLayer(layer_h, nonlinearity=rectify)

        layer_h_to_h = NonlinearityLayer(InputLayer((None, DIMENSION_HIDDEN_LAYERS))
                                                        , nonlinearity=rectify)
        # assert ""
        layer_social_lstm = CustomRecurrentLayer(layer_in, layer_h, layer_h_to_h
                                                                , nonlinearity=None, hid_init=Constant(.0))
        temp_outputs = get_output(layer_social_lstm, net_inputs)
        temp_outputs = get_output(layer_h, {layer_in: net_inputs, layer_prev_h: layer_h_to_h.get_output_for(layer_in)})

        net_outputs = temp_outputs


    except KeyboardInterrupt:
        pass


# def test_model():

    # all_samples, all_targets = sample.load_batch_for_nodes(sample.read_traces_from_path(PATH_TRACE_FILES),
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
    # updates = lasagne.updates.rmsprop(cost, all_params, LEARNING_RATE_RMSPROP)
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
    #     ins, tars = sample.load_batch_for_nodes(all_traces, 1, [], 0, True)
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
    #                 inputs, targets = sample.load_batch_for_nodes(all_traces, SIZE_BATCH, [], p, True)
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


if __name__ == '__main__':
    test_model()
