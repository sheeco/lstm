# coding:utf-8

PATH_TRACE_FILES = 'res/trace/'
# Length of observed sequence
LENGTH_SEQUENCE_INPUT = 10
# Length of predicted sequence
LENGTH_SEQUENCE_OUTPUT = 1
# Number of units in each hidden (LSTM) layers
DIMENSION_HIDDEN_LAYERS = [3, 3]
# Optimization learning rate
LEARNING_RATE = .05
# All gradients above this will be clipped
GRAD_CLIP = 100
# How often should we check the output
LOG_SLOT = 100
# Number of epochs to train the net
NUM_EPOCH = 100
# Batch Size
SIZE_BATCH = 50
# 2 for [x, y]; 3 for [sec, x, y]
DIMENSION_INPUT = 2
