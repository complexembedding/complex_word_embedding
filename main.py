import sys
import os
import numpy as np
sys.path.append('complexnn')

from keras.models import Model, Input
from embedding import phase_embedding_layer, amplitude_embedding_layer
from mat_multiply import complex_multiply
from data import orthonormalized_word_embeddings,get_lookup_table
from data_reader import SSTDataReader


def run_complex_embedding_network(lookup_table, max_sequence_length):

    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')

    amplitude_embedding = amplitude_embedding_layer(lookup_table, max_sequence_length)(sequence_input)

    phase_embedding = phase_embedding_layer(max_sequence_length, lookup_table.shape[0])(sequence_input)

    output = complex_multiply()([phase_embedding, amplitude_embedding])

    model = Model(sequence_input, output)
    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
    return model

def main():
    dir_name = 'C:/Users/quartz/Documents/python/complex_word_embedding/'
    path_to_vec = 'glove/glove.6B.100d.txt'#
    reader = SSTDataReader(dir_name,nclasses = 2)
    embedding_params = reader.get_word_embedding(path_to_vec)
    lookup_table = get_lookup_table(embedding_params)
    # print(lookup_table.shape)
    # print(len(embedding_params['word2id']))
    model = run_complex_embedding_network(lookup_table, 10)
    model.summary()
    # word2id = embedding_params['word2id']
    # word_vec = embedding_params['word_vec']
     # params = {'word2id':word2id, 'word_vec':word_vec, 'wvec_dim':wvec_dim,'word_complex_phase':word_complex_phase}



    # max_sequence_length = 10
    # model = run_complex_embedding_network(path_to_vec, max_sequence_length)


if __name__ == '__main__':
    main()
