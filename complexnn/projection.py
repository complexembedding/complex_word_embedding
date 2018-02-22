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
from keras.constraints import UnitNorm
from embedding import phase_embedding_layer, amplitude_embedding_layer
import keras.backend as K
import math

class complex_projection(Layer):

    def __init__(self, dimension, **kwargs):
        # self.output_dim = output_dim
        super(complex_projection, self).__init__(**kwargs)
        self.dimension = dimension

    def build(self, input_shape):

        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.dimension,self.dimension,2),
                                      constraint = None,
                                      initializer='uniform',
                                      trainable=True)
        # Create a trainable weight variable for this layer.

        # if len(input_shape) != 1:
        #     raise ValueError('This layer should be called '
        #                      'on a only one input. '
        #                      'Got ' + str(len(input_shape)) + ' inputs.')


        # self.kernel = self.add_weight(name='kernel',
        #                               shape=(input_shape[1], self.output_dim),
        #                               initializer='uniform',
        #                               trainable=True)
        super(complex_projection, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):

        # if len(inputs) != 1:
        #     raise ValueError('This layer should be called '
        #                      'on only 1 input.'
        #                      'Got ' + str(len(input)) + ' inputs.')

        #Implementation of Tr(P|v><v|) = ||P|v>||^2
        P_real = self.kernel[:,:,0]
        P_imag = self.kernel[:,:,1]

        v_real = inputs[:,:,0]
        v_imag = inputs[:,:,1]

        # print(P_real.shape)
        # print(v_real.shape)
        # print(K.sum(K.dot(P_real,K.transpose(v_real)),axis = 0))
        Pv_real = K.dot(P_real,K.transpose(v_real))-K.dot(P_imag,K.transpose(v_imag))
        Pv_imag = K.dot(P_real,K.transpose(v_imag))+K.dot(P_imag,K.transpose(v_real))
        y = K.sum(K.square(Pv_real), axis = 0)+K.sum(K.square(Pv_imag), axis = 0)
        # print(y)
        return y

    def compute_output_shape(self, input_shape):
        # print(type(input_shape[1]))
        output_shape = [None,2]
        return([tuple(output_shape)])

class complex_1d_projection(Layer):

    def __init__(self, dimension, **kwargs):
        # self.output_dim = output_dim
        super(complex_1d_projection, self).__init__(**kwargs)
        self.dimension = dimension

    def build(self, input_shape):

        self.kernel = self.add_weight(name='kernel',
                                      shape=(1,self.dimension,2),
                                      constraint = UnitNorm(axis = [1,2]),
                                      initializer='uniform',
                                      trainable=True)
        # Create a trainable weight variable for this layer.

        # if len(input_shape) != 1:
        #     raise ValueError('This layer should be called '
        #                      'on a only one input. '
        #                      'Got ' + str(len(input_shape)) + ' inputs.')


        # self.kernel = self.add_weight(name='kernel',
        #                               shape=(input_shape[1], self.output_dim),
        #                               initializer='uniform',
        #                               trainable=True)
        super(complex_1d_projection, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):

        # if len(inputs) != 1:
        #     raise ValueError('This layer should be called '
        #                      'on only 1 input.'
        #                      'Got ' + str(len(input)) + ' inputs.')

        #Implementation of ||<p|v>||^2

        P_real = self.kernel[:,:,0]
        P_imag = self.kernel[:,:,1]

        v_real = inputs[:,:,0]
        v_imag = inputs[:,:,1]

        print(P_real.shape)
        print(P_imag.shape)
        print(v_real.shape)
        print(v_imag.shape)
        # print(K.sum(K.dot(P_real,K.transpose(v_real)),axis = 0))
        Pv_real = K.dot(P_real,K.transpose(v_real))-K.dot(P_imag,K.transpose(v_imag))
        Pv_imag = K.dot(P_real,K.transpose(v_imag))+K.dot(P_imag,K.transpose(v_real))

        print(Pv_real.shape)
        print(Pv_imag.shape)
        y = K.square(Pv_real)+K.square(Pv_imag)
        # y = K.sum(K.square(v_real))
        return y

    def compute_output_shape(self, input_shape):
        # print(type(input_shape[1]))
        output_shape = [None,1]
        return([tuple(output_shape)])



def main():

    sequence_input = Input(shape=(10,2), dtype='float')
    output = complex_1d_projection(10)(sequence_input)
    model = Model(sequence_input, output)

    model.compile(loss='mean_squared_error',
              optimizer='rmsprop',
              metrics=['acc'])
    model.summary()

    # for i in range(1000):
    x = np.random.random((1,10,2))
    # y = K.sum(K.square(x), axis=None, keepdims = False)
    x = x/np.linalg.norm(x, ord = 2, axis = (1,2))
    # print(np.linalg.norm(x[0], ord = 2))
        # # print(np.linalg.norm(x))
    y = model.predict(x)
    model.fit(x,y)
    for i in range(100):
        x = np.random.random((1,10,2))
        x = x/np.linalg.norm(x, ord = 2, axis = (1,2))
        y = model.predict(x)
        print(y)


if __name__ == '__main__':
    main()
