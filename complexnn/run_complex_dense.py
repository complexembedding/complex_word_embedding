from utils import *
from dense import ComplexDense
# from conv1D import *
import numpy as np
from keras import backend as K
from keras import activations, initializers, regularizers, constraints
from keras.layers import Layer, InputSpec
from keras.models import Model, Input
from keras.initializers import Constant
from keras.layers.convolutional import (
    Convolution2D, Convolution1D, MaxPooling1D, AveragePooling1D)
from keras.layers.core import Permute
from keras.layers.core import Dense, Activation, Flatten
import tensorflow as tf
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import to_categorical

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

def one_hidden_layer_complex_nn(input_size = 300, output_size = 2):
    inputs = Input(shape = (input_size,))

    # conv = ComplexConv1D(
    #     32, 512, strides=16,
    #     activation='relu')(inputs)
    # pool = AveragePooling1D(pool_size=4, strides=2)(conv)

    # pool = Permute([2, 1])(pool)
    # flattened = Flatten()(pool)

    predictions = ComplexDense(units = output_size, activation='sigmoid', bias_initializer=Constant(value=-5))(inputs)
    # predictions = ComplexDense(
    #     output_size,
    #     activation='sigmoid',
    #     bias_initializer=Constant(value=-5))(dense)

    predictions = GetReal()(predictions)
    model = Model(inputs=inputs, outputs=predictions)

    model.compile(optimizer=Adam(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# def main(model_name, model, local_data, epochs, fourier,
#          stft, fast_load):

def main():
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

    print(".. building model")
    # model = get_shallow_convnet(window_size=4096, channels=2, output_size=84)
    model = one_hidden_layer_complex_nn(input_size = 300, output_size = 2)
    model.summary()
    print(".. parameters: {:03.2f}M".format(model.count_params() / 1000000.))


    # x =
    x = np.random.random((1,300))
    y = to_categorical(np.random.randint(2, size=(1, 1)), num_classes=2)


    for i in range(700):
        model.fit(x,y)

    print(y)
    print(model.predict(x))
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
