import sys
import os
import numpy as np
import codecs
sys.path.append('complexnn')

from keras.models import Model, Input, model_from_json
from embedding import phase_embedding_layer, amplitude_embedding_layer
from mat_multiply import complex_multiply
from data import orthonormalized_word_embeddings,get_lookup_table, batch_gen,data_gen
from data_reader import SSTDataReader
from average import complex_average
from keras.preprocessing.sequence import pad_sequences
from projection import complex_projection, complex_1d_projection
from keras.utils import to_categorical
import matplotlib.pyplot as plt

def run_complex_embedding_network(lookup_table, max_sequence_length):

    embedding_dimension = lookup_table.shape[1]
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')

    phase_embedding = phase_embedding_layer(max_sequence_length, lookup_table.shape[0])(sequence_input)


    amplitude_embedding = amplitude_embedding_layer(np.transpose(lookup_table), max_sequence_length)(sequence_input)

    sentence_embedding_seq = complex_multiply()([phase_embedding, amplitude_embedding])


    avg = complex_average()(sentence_embedding_seq)

    output = complex_1d_projection(dimension = embedding_dimension)(avg)


    model = Model(sequence_input, output)
    model.compile(loss='mean_squared_error',
              optimizer='rmsprop',
              metrics=['acc'])
    return model

def main():
    dir_name = 'C:/Users/quartz/Documents/python/complex_word_embedding/'
    path_to_vec = 'glove/glove.6B.100d.txt'#


    # model = load_model('model/model_1', 'model/weight_1')

    reader = SSTDataReader(dir_name,nclasses = 2)
    embedding_params = reader.get_word_embedding(path_to_vec)
    lookup_table = get_lookup_table(embedding_params)
    max_sequence_length = 60


    model = run_complex_embedding_network(lookup_table, max_sequence_length)
    model.summary()

    #################################################################
    # # Training

    # -1 refers to loading the whole data at once instead of in mini-batches
    train_test_val= reader.create_batch(embedding_params = embedding_params,batch_size = -1)

    training_data = train_test_val['train']
    test_data = train_test_val['test']
    validation_data = train_test_val['dev']


    # for x, y in batch_gen(training_data, max_sequence_length):
    #     model.train_on_batch(x,y)

    train_x, train_y = data_gen(training_data, max_sequence_length)
    test_x, test_y = data_gen(test_data, max_sequence_length)
    val_x, val_y = data_gen(validation_data, max_sequence_length)

    assert len(train_x) == 67349
    assert len(test_x) == 1821
    assert len(val_x) == 872

    history = model.fit(x=train_x, y = train_y, batch_size = 32, epochs= 1,validation_data= (val_x, val_y))


    val_acc= history.history['val_acc']
    train_acc = history.history['acc']
    # print(val_perf)
    # print(train_perf)

    line_1, = plt.plot(val_acc)
    line_2, = plt.plot(train_acc)
    # plt.axis([0, 6, 0, 20])

    plt.legend([line_1, line_2], ['val_acc', 'train_acc'])
    plt.show()
    plt.savefig('learning_curve.png')


    evaluation = model.evaluate(x = test_x, y = test_y)
    print(evaluation)



    save_model_structure(model, 'model/model_1')
    save_model_weights(model, 'model/weight_1')
    #################################################################

def save_model_structure(model, model_structure_path):
    json_string = model.to_json()
    data_out = codecs.open(model_structure_path,'w')
    data_out.write(json_string)
    data_out.close()

def save_model_weights(model, model_weights_path):
    model.save_weights(model_weights_path)

def load_model_structure(model_structure_path):
    data_in = codecs.open(model_structure_path)
    json_string = data_in.read()
    model = model_from_json(json_string)
    data_in.close()
    return model

def load_model_weights(model, model_weights_path):
    model.load_weights(model_weights_path)
    return model

def load_model(model_structure_path, model_weights_path):
    model = load_model_structure(model_structure_path)
    load_model_weights(model, model_weights_path)
    return model


if __name__ == '__main__':
    main()
