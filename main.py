import sys
import os
import numpy as np
sys.path.append('complexnn')

from keras.models import Model, Input
from embedding import phase_embedding_layer, amplitude_embedding_layer
from mat_multiply import complex_multiply
from data import orthonormalized_word_embeddings,get_lookup_table, batch_gen
from data_reader import SSTDataReader
from average import complex_average
from keras.preprocessing.sequence import pad_sequences
from projection import complex_projection
from keras.utils import to_categorical


def run_complex_embedding_network(lookup_table, max_sequence_length):

    embedding_dimension = lookup_table.shape[1]
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')

    phase_embedding = phase_embedding_layer(max_sequence_length, lookup_table.shape[0])(sequence_input)


    amplitude_embedding = amplitude_embedding_layer(np.transpose(lookup_table), max_sequence_length)(sequence_input)

    sentence_embedding_seq = complex_multiply()([phase_embedding, amplitude_embedding])


    avg = complex_average()(sentence_embedding_seq)

    output = complex_projection(dimension = embedding_dimension)(avg)


    model = Model(sequence_input, output)
    model.compile(loss='mean_squared_error',
              optimizer='rmsprop',
              metrics=['acc'])
    return model

def main():
    dir_name = 'C:/Users/quartz/Documents/python/complex_word_embedding/'
    path_to_vec = 'glove/glove.6B.100d.txt'#
    reader = SSTDataReader(dir_name,nclasses = 2)
    embedding_params = reader.get_word_embedding(path_to_vec)
    lookup_table = get_lookup_table(embedding_params)
    max_sequence_length = 60


    model = run_complex_embedding_network(lookup_table, max_sequence_length)
    model.summary()

    #################################################################
    # Training
    sentences = reader.create_batch(embedding_params = embedding_params,batch_size = 4)
    training_data = sentences['train']
    for x, y in batch_gen(training_data, max_sequence_length):
        # print(x,y)
        # y_binary = to_categorical(y)
        model.train_on_batch(x,y)
        # print(x)
        # y0 = model.predict(x)
        # print(y0)
        # print(type(y0))
        # print(y0.shape)
        # print(y)
        # print(type(y))
        # print(y.shape)

    #################################################################


if __name__ == '__main__':
    main()
