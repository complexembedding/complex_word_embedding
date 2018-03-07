from dense import ComplexDense
import numpy as np
from keras import backend as K
from keras.layers import Layer
from keras.models import Model, Input
import tensorflow as tf
import sys
import os
import keras.backend as K
import math


class ComplexMixture(Layer):

    def __init__(self, **kwargs):
        # self.output_dim = output_dim
        super(ComplexMixture, self).__init__(**kwargs)


    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')

        if len(input_shape) != 2:
             raise ValueError('This layer should be called '
                             'on a list of 2 inputs.'
                              'Got ' + str(len(input_shape)) + ' inputs.')

        super(ComplexMixture, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):

        if not isinstance(inputs, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')

        if len(inputs) != 2:
            raise ValueError('This layer should be called '
                            'on a list of 2 inputs.'
                            'Got ' + str(len(inputs)) + ' inputs.')



        input_real = K.expand_dims(inputs[0]) #shape: (None, 60, 300, 1)
        input_imag = K.expand_dims(inputs[1]) #shape: (None, 60, 300, 1)


        input_real_transpose = K.permute_dimensions(input_real, (0,1,3,2)) #shape: (None, 60, 1, 300)

        input_imag_transpose = K.permute_dimensions(input_imag, (0,1,3,2)) #shape: (None, 60, 1, 300)

        ###
        #output = (input_real+i*input_imag)(input_real_transpose-i*input_imag_transpose)
        ###
        output_real = K.batch_dot(input_real_transpose,input_real, axes = [2,3])+ K.batch_dot(input_imag_transpose, input_imag,axes = [2,3])  #shape: (None, 60, 300, 300)

        output_imag = K.batch_dot(input_real_transpose, input_imag,axes = [2,3])-K.batch_dot(input_imag_transpose,input_real, axes = [2,3])  #shape: (None, 60, 300, 300)

        output_real = K.mean(output_real, axis = 1, keepdims = False) #shape: (None, 300, 300)

        output_imag = K.mean(output_imag, axis = 1, keepdims = False) #shape: (None, 300, 300)


        # output_real = K.mean(input_real,axis = 1, keepdims = False)
        # output_imag = K.mean(input_imag,axis = 1, keepdims = False)
        print(output_real.shape)
        print(output_imag.shape)
        # print(y.shape)
        return [output_real, output_imag]

    def compute_output_shape(self, input_shape):
        # print(type(input_shape[1]))
        one_input_shape = list(input_shape[0])
        one_output_shape = [one_input_shape[0], one_input_shape[2], one_input_shape[2]]
        return [tuple(one_output_shape), tuple(one_output_shape)]


def main():
    input_2 = Input(shape=(3,5), dtype='float')
    input_1 = Input(shape=(3,5), dtype='float')
    [output_1, output_2] = ComplexMixture()([input_1, input_2])


    model = Model([input_1, input_2], [output_1, output_2])
    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    model.summary()

    x = np.random.random((3,3,5))
    x_2 = np.random.random((3,3,5))


    print(x)
    print(x_2)
    output = model.predict([x,x_2])
    print(output)

if __name__ == '__main__':
    main()
