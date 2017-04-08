# coding:utf-8

# Path for trace files
PATH_TRACE_FILES = 'res/trace/'
# Path for logs
PATH_LOG = 'log/'
# How often should we check the output
LOG_SLOT = 1
# Show warning info or not
SHOW_WARNING = False
# Level of printing detail
# 0 means mandatory printing only
PRINT_LEVEL = 3
# # How many nodes to learn on
# N_NODES_EXPECTED = 3
# Dimension of sample input: 2 for [x, y]; 3 for [sec, x, y]
DIMENSION_SAMPLE = 3
# If only a full size batch will be accepted
STRICT_BATCH_SIZE = True
# Length of edge when coordinates need to be mapped to grid
GRAIN_GRID = 100

if __debug__:

    """Nano-size net config for debugging"""
    # Length of observed sequence
    LENGTH_SEQUENCE_INPUT = 4
    # Length of predicted sequence
    LENGTH_SEQUENCE_OUTPUT = 4
    # Number of units in embedding layer ([x, y] -> e)
    DIMENSION_EMBED_LAYER = 2
    # Number of units in hidden (LSTM) layers
    # n: single layer of n units
    # (m, n): m * layers of n units
    DIMENSION_HIDDEN_LAYERS = (2, 2)

    # # Social-LSTM config
    #
    # # x * x of arean for each pool
    # SIZE_POOL = 100
    # # x * x of pools around each node
    # RANGE_NEIGHBORHOOD = 4

    # Batch Size
    SIZE_BATCH = 10
    # Number of epochs to train the net
    NUM_EPOCH = 2
    # Optimization learning rate
    LEARNING_RATE_RMSPROP = .03
    RHO_RMSPROP = .9
    EPSILON_RMSPROP = 1e-8
    # All gradients above this will be clipped
    GRAD_CLIP = 100

else:

    # Length of observed sequence
    LENGTH_SEQUENCE_INPUT = 10
    # Length of predicted sequence
    LENGTH_SEQUENCE_OUTPUT = 10
    # Number of units in embedding layer ([x, y] -> e)
    DIMENSION_EMBED_LAYER = 64
    # Number of units in each hidden (LSTM) layers
    DIMENSION_HIDDEN_LAYERS = (10, 64)

    # # # Social-LSTM config
    #
    # # x * x of arean for each pool
    # SIZE_POOL = 100
    # # x * x of pools around each node
    # RANGE_NEIGHBORHOOD = 4

    # Batch Size
    SIZE_BATCH = 10
    # Number of epochs to train the net
    NUM_EPOCH = 3
    # Optimization learning rate
    LEARNING_RATE_RMSPROP = .001
    RHO_RMSPROP = .95
    EPSILON_RMSPROP = 1e-8
    # All gradients above this will be clipped
    GRAD_CLIP = 0
