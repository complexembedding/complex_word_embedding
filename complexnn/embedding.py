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
import keras.backend as K
import math

def phase_embedding_layer(max_sequence_length, input_dim):
    embedding_layer = Embedding(input_dim,
                            1,
                            embeddings_initializer=RandomUniform(minval=0, maxval=2*math.pi),
                            input_length=max_sequence_length)
    return embedding_layer


def amplitude_embedding_layer(embedding_matrix, max_sequence_length):
    embedding_dim = embedding_matrix.shape[0]
    vocabulary_size = embedding_matrix.shape[1]
    embedding_layer = Embedding(vocabulary_size,
                            embedding_dim,
                            weights=[np.transpose(embedding_matrix)],
                            input_length=max_sequence_length,
                            trainable=False)

    return embedding_layer


# def amplitude_embedding_layer()

# def get_shallow_convnet(window_size=4096, channels=2, output_size=84):
#     inputs = Input(shape=(window_size, channels), dtype = tf.float32)

#     conv = ComplexConv1D(
#         32, 512, strides=16,
#         activation='relu')(inputs)
#     pool = AveragePooling1D(pool_size=4, strides=2)(conv)

#     pool = Permute([2, 1])(pool)
#     flattened = Flatten()(pool)

#     dense = ComplexDense(2048, activation='relu')(flattened)
#     # dense = ComplexDense(2048, activation='relu')(inputs)
#     predictions = ComplexDense(
#         output_size,
#         activation='sigmoid',
#         bias_initializer=Constant(value=-5))(dense)
#     predictions = GetReal()(predictions)
#     model = Model(inputs=inputs, outputs=predictions)

#     model.compile(optimizer=Adam(lr=1e-4),
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])
#     return model



# def main(model_name, model, local_data, epochs, fourier,
#          stft, fast_load):

def main():
    path_to_vec = '../glove/glove.6B.300d.txt'
    embedding_matrix, word_list = orthonormalized_word_embeddings(path_to_vec)
    max_sequence_length = 10
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    phase_embedding = phase_embedding_layer(max_sequence_length, len(word_list))
    output = phase_embedding(sequence_input)
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

    x = np.array([[1,2,3,4,5,6,7,8,9,10]])
    print(model.predict(x))

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
