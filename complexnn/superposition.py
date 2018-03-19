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


class ComplexSuperposition(Layer):

    def __init__(self, **kwargs):
        # self.output_dim = output_dim
        self. trainable = False
        super(ComplexSuperposition, self).__init__(**kwargs)

    def get_config(self):
        config = {'trainable': self.trainable}
        base_config = super(ComplexSuperposition, self).get_config()
        return dict(list(base_config.items())+list(config.items()))

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')

        if len(input_shape) != 2:
             raise ValueError('This layer should be called '
                             'on a list of 2 inputs.'
                              'Got ' + str(len(input_shape)) + ' inputs.')

        super(ComplexSuperposition, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):

        if not isinstance(inputs, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')

        if len(inputs) != 2:
            raise ValueError('This layer should be called '
                            'on a list of 2 inputs.'
                            'Got ' + str(len(inputs)) + ' inputs.')


        # if len(inputs) != 1:
        #     raise ValueError('This layer should be called '
        #                      'on only 1 input.'
        #                      'Got ' + str(len(input)) + ' inputs.')
        input_real = inputs[0]
        input_imag = inputs[1]

        output_real = K.mean(input_real,axis = 1, keepdims = False)
        output_imag = K.mean(input_imag,axis = 1, keepdims = False)

        # print(y.shape)
        return [output_real, output_imag]

    def compute_output_shape(self, input_shape):
        # print(type(input_shape[1]))
        one_input_shape = list(input_shape[0])
        one_output_shape = [one_input_shape[0], one_input_shape[2]]
        return [tuple(one_output_shape), tuple(one_output_shape)]



def main():
    input_2 = Input(shape=(3,5), dtype='float')
    input_1 = Input(shape=(3,5), dtype='float')
    [output_1, output_2] = ComplexSuperposition()([input_1, input_2])


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
