# coding:utf-8

PATH_TRACE_FILES = 'res/trace/'
N_NODES = 2
# Dimension of sample input: 2 for [x, y]; 3 for [sec, x, y]
DIMENSION_SAMPLE = 2
# Length of observed sequence
LENGTH_SEQUENCE_INPUT = 10
# Length of predicted sequence
LENGTH_SEQUENCE_OUTPUT = 1
# Number of units in embedding layer ([x, y] -> e)
# DIMENSION_EMBED_LAYER = 64
DIMENSION_EMBED_LAYER = 3
# Number of units in each hidden (LSTM) layers
# DIMENSION_HIDDEN_LAYERS = 128
DIMENSION_HIDDEN_LAYERS = 5
# Number of hidden (LSTM) layers
N_HIDDEN_LAYERS = 1
# x * x of arean for each pool
SIZE_POOL = 10
# x * x of pools around each node
# RANGE_NEIGHBORHOOD = 32
RANGE_NEIGHBORHOOD = 10
# Optimization learning rate
LEARNING_RATE_RMSPROP = .003
GAMMA_RMSPROP = .9
EPSILON_RMSPROP = 1e-8
# All gradients above this will be clipped
GRAD_CLIP = 100
# Batch Size
SIZE_BATCH = 50
# Number of epochs to train the net
NUM_EPOCH = 100
# How often should we check the output
LOG_SLOT = 100
