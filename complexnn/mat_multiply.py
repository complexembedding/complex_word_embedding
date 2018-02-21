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


class complex_multiply(Layer):

    def __init__(self, **kwargs):
        # self.output_dim = output_dim
        super(complex_multiply, self).__init__(**kwargs)


    def build(self, input_shape):
        # Create a trainable weight variable for this layer.

        if not isinstance(input_shape, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')

        if len(input_shape) != 2:
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs. '
                             'Got ' + str(len(input_shape)) + ' inputs.')


        # self.kernel = self.add_weight(name='kernel',
        #                               shape=(input_shape[1], self.output_dim),
        #                               initializer='uniform',
        #                               trainable=True)
        super(complex_multiply, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')

        if len(inputs) != 2:
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.'
                             'Got ' + str(len(input)) + ' inputs.')

        phase = inputs[0]
        amplitude = inputs[1]

        sentence_length = amplitude.shape[1]
        embedding_dim = amplitude.shape[2]


        real_part = K.repeat_elements(K.cos(phase), embedding_dim, axis = 2)*amplitude
        imag_part = K.repeat_elements(K.sin(phase), embedding_dim, axis = 2)*amplitude
        # print(real_part.shape)
        # print(imag_part.shape)

        real_part = K.reshape(real_part,[-1,sentence_length,embedding_dim,1])
        imag_part =  K.reshape(imag_part,[-1,sentence_length,embedding_dim,1])
        y = K.concatenate([real_part,imag_part],axis = -1)
        # print(y.shape)
        return y

    def compute_output_shape(self, input_shape):
        # print(type(input_shape[1]))
        output_shape = list(input_shape[1])
        output_shape.append(2)
        return(tuple(output_shape))

def main():
    path_to_vec = '../glove/glove.6B.100d.txt'
    embedding_matrix, word_list = orthonormalized_word_embeddings(path_to_vec)
    max_sequence_length = 10
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')

    amplitude_embedding = amplitude_embedding_layer(embedding_matrix,max_sequence_length)(sequence_input)

    phase_embedding = phase_embedding_layer(max_sequence_length, len(word_list))(sequence_input)

    output = complex_multiply()([phase_embedding, amplitude_embedding])

    model = Model(sequence_input, output)
    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
    model.summary()

    # sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    # path_to_vec = '../glove/glove.6B.100d.txt'
    # embedded_sequences = amplitude_embedding_layer(path_to_vec, 10)

    # output = embedded_sequences(sequence_input)
    # model = Model(sequence_input, output)
    # model.compile(loss='categorical_crossentropy',
    #           optimizer='rmsprop',
    #           metrics=['acc'])

    # model.summary()

    # x = np.array([[0,2,3,4,5,6,7,8,9,10]])
    # y = model.predict(x)
    # print(y)
    # print(y.shape)

    # rng = numpy.random.RandomState(123)

    # Warning: the full dataset is over 40GB. Make sure you have enough RAM!
    # This can take a few minutes to load
    # if in_memory:
    #     print('.. loading train data')
    #     dataset = MusicNet(local_data, complex_=complex_, fourier=fourier,
    #                        stft=stft, rng=rng, fast_load=fast_load)
    #     dataset.load()
    #     print('.. train data loaded')
    #     Xvalid, Yvalid = dataset.eval_set('valid')
    #     Xtest, Ytest = dataset.eval_set('test')
    # else:
    #     raise ValueError

    # print(".. building model")
    # # model = get_shallow_convnet(window_size=4096, channels=2, output_size=84)
    # model = one_hidden_layer_complex_nn(input_size = 300, output_size = 2)
    # model.summary()
    # print(".. parameters: {:03.2f}M".format(model.count_params() / 1000000.))


    # # x =
    # x = np.random.random((1,300))
    # y = to_categorical(np.random.randint(2, size=(1, 1)), num_classes=2)


    # for i in range(700):
    #     model.fit(x,y)

    # print(y)
    # print(model.predict(x))
    # if in_memory:
    #     pass
    #     # do nothing
    # else:
    #     raise ValueError

    # logger = mimir.Logger(
    #     filename='models/log_{}.jsonl.gz'.format(model_name))

    # it = dataset.train_iterator()

    # callbacks = [Validation(Xvalid, Yvalid, 'valid', logger),
    #              Validation(Xtest, Ytest, 'test', logger),
    #              SaveLastModel("./models/", 1, name=model),
    #              Performance(logger),
    #              LearningRateScheduler(schedule)]

    # print('.. start training')
    # model.fit_generator(
    #     it, steps_per_epoch=1000, epochs=epochs,
    #     callbacks=callbacks, workers=1)

if __name__ == '__main__':
    main()
