# coding:utf-8

# Path for trace files
PATH_TRACE_FILES = 'res/trace/'
# How many nodes to learn on
N_NODES_EXPECTED = 9
# How often should we check the output
LOG_SLOT = 100

if __debug__:

    """Nano-size net config for debugging"""
    # Batch Size
    SIZE_BATCH = 3
    # If only a full size batch will be accepted
    STRICT_BATCH_SIZE = True
    # Dimension of sample input: 2 for [x, y]; 3 for [sec, x, y]
    DIMENSION_SAMPLE = 3
    # Length of observed sequence
    LENGTH_SEQUENCE_INPUT = 4
    # Length of predicted sequence
    LENGTH_SEQUENCE_OUTPUT = 4
    # Number of units in embedding layer ([x, y] -> e)
    DIMENSION_EMBED_LAYER = 2
    # Number of units in each hidden (LSTM) layers
    DIMENSION_HIDDEN_LAYERS = 2
    # Number of hidden (LSTM) layers
    N_HIDDEN_LAYERS = 1
    # x * x of arean for each pool
    SIZE_POOL = 100
    # x * x of pools around each node
    RANGE_NEIGHBORHOOD = 4
    # Optimization learning rate
    LEARNING_RATE_RMSPROP = .03
    GAMMA_RMSPROP = .9
    EPSILON_RMSPROP = 1e-8
    # All gradients above this will be clipped
    GRAD_CLIP = 100
    # Number of epochs to train the net
    NUM_EPOCH = 100
    SHOW_WARNING = False

else:

    # Batch Size
    SIZE_BATCH = 10
    # If only a full size batch will be accepted
    STRICT_BATCH_SIZE = True
    # Dimension of sample input: 2 for [x, y]; 3 for [sec, x, y]
    DIMENSION_SAMPLE = 3
    # Length of observed sequence
    LENGTH_SEQUENCE_INPUT = 10
    # Length of predicted sequence
    LENGTH_SEQUENCE_OUTPUT = 10
    # Number of units in embedding layer ([x, y] -> e)
    DIMENSION_EMBED_LAYER = 64
    # Number of units in each hidden (LSTM) layers
    DIMENSION_HIDDEN_LAYERS = 100
    # Number of hidden (LSTM) layers
    N_HIDDEN_LAYERS = 1
    # x * x of arean for each pool
    SIZE_POOL = 100
    # x * x of pools around each node
    RANGE_NEIGHBORHOOD = 4
    # Optimization learning rate
    LEARNING_RATE_RMSPROP = .03
    GAMMA_RMSPROP = .9
    EPSILON_RMSPROP = 1e-8
    # All gradients above this will be clipped
    GRAD_CLIP = 100
    # Number of epochs to train the net
    NUM_EPOCH = 100
    SHOW_WARNING = False
