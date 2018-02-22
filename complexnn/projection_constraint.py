from utils import *
from dense import ComplexDense
import numpy as np
from keras import backend as K
from keras import activations, initializers, regularizers, constraints
from keras.layers import Layer, InputSpec
from keras.models import Model, Input
from keras.initializers import Constant, RandomUniform
from keras.layers.convolutional import (
    Convolution2D, Convolution1D, MaxPooling1D, AveragePooling1D)
from keras.layers.core import Permute
from keras.layers.core import Dense, Activation, Flatten
import tensorflow as tf
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import to_categorical
from keras.layers import Embedding
from data import *
from keras.constraints import Constraint
from embedding import phase_embedding_layer, amplitude_embedding_layer
import keras.backend as K
import math


class ProjectionConstraint(Constraint):

    # def __init__(self):
    #     self.output_dim = output_dim
    #     super(complex_projection, self).__init__(**kwargs)

    def __call__(self, w):

        # if len(inputs) != 1:
        #     raise ValueError('This layer should be called '
        #                      'on only 1 input.'
        #                      'Got ' + str(len(input)) + ' inputs.')
        print(inputs.shape)
        y = K.mean(inputs,axis = 1)
        # print(y.shape)
        return y


