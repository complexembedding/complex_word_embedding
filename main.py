import sys
import os
import numpy as np
import codecs
sys.path.append('complexnn')

from keras.models import Model, Input, model_from_json, load_model
from keras.layers import Embedding, GlobalAveragePooling1D,Dense, Masking, Flatten
from embedding import phase_embedding_layer, amplitude_embedding_layer
from multiply import ComplexMultiply
from data import orthonormalized_word_embeddings,get_lookup_table, batch_gen,data_gen
from mixture import ComplexMixture
from data_reader import *
from superposition import ComplexSuperposition
from keras.preprocessing.sequence import pad_sequences
from projection import Complex1DProjection
from keras.utils import to_categorical
from keras.constraints import unit_norm
from dense import ComplexDense
from utils import GetReal
from keras.initializers import Constant
from params import Params
import matplotlib.pyplot as plt

def run_complex_embedding_network_2(lookup_table, max_sequence_length, nb_classes = 2):

    embedding_dimension = lookup_table.shape[1]
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')

    phase_embedding = phase_embedding_layer(max_sequence_length, lookup_table.shape[0], embedding_dimension, trainable = True)(sequence_input)


    amplitude_embedding = amplitude_embedding_layer(np.transpose(lookup_table), max_sequence_length, trainable = True)(sequence_input)

    [seq_embedding_real, seq_embedding_imag] = ComplexMultiply()([phase_embedding, amplitude_embedding])


    [sentence_embedding_real, sentence_embedding_imag]= ComplexMixture()([seq_embedding_real, seq_embedding_imag])

    sentence_embedding_real = Flatten()(sentence_embedding_real)
    sentence_embedding_imag = Flatten()(sentence_embedding_imag)
    # output = Complex1DProjection(dimension = embedding_dimension)([sentence_embedding_real, sentence_embedding_imag])
    predictions = ComplexDense(units = nb_classes, activation='sigmoid', bias_initializer=Constant(value=-1))([sentence_embedding_real, sentence_embedding_imag])

    output = GetReal()(predictions)

    model = Model(sequence_input, output)
    return model




def run_complex_embedding_network(lookup_table, max_sequence_length, nb_classes = 2):

    embedding_dimension = lookup_table.shape[1]
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')

    phase_embedding = phase_embedding_layer(max_sequence_length, lookup_table.shape[0])(sequence_input)


    amplitude_embedding = amplitude_embedding_layer(np.transpose(lookup_table), max_sequence_length, trainable = True)(sequence_input)

    [seq_embedding_real, seq_embedding_imag] = ComplexMultiply()([phase_embedding, amplitude_embedding])


    [sentence_embedding_real, sentence_embedding_imag]= ComplexSuperposition()([seq_embedding_real, seq_embedding_imag])

    # output = Complex1DProjection(dimension = embedding_dimension)([sentence_embedding_real, sentence_embedding_imag])
    predictions = ComplexDense(units = nb_classes, activation='sigmoid', bias_initializer=Constant(value=-1))([sentence_embedding_real, sentence_embedding_imag])

    output = GetReal()(predictions)

    model = Model(sequence_input, output)
    model.compile(loss='binary_crossentropy',
          optimizer='rmsprop',
          metrics=['accuracy'])
    return model

def run_real_network(lookup_table, max_sequence_length):
    embedding_dimension = lookup_table.shape[1]
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedding = Embedding(trainable=True, input_dim=lookup_table.shape[0],output_dim=lookup_table.shape[1], weights=[lookup_table],embeddings_constraint = unit_norm(axis = 1))(sequence_input)
    representation =GlobalAveragePooling1D()(embedding)
    output=Dense(1, activation='sigmoid')(representation)

    model = Model(sequence_input, output)
    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    return model

# def save_model_structure(model, model_structure_path):
#     json_string = model.to_json()
#     data_out = codecs.open(model_structure_path,'w')
#     data_out.write(json_string)
#     data_out.close()

# def save_model_weights(model, model_weights_path):
#     model.save_weights(model_weights_path)

# def load_model_structure(model_structure_path):
#     data_in = codecs.open(model_structure_path)
#     json_string = data_in.read()
#     model = model_from_json(json_string)
#     data_in.close()
#     return model

# def load_model_weights(model, model_weights_path):
#     model.load_weights(model_weights_path)
#     return model

# def load_model(model_structure_path, model_weights_path):
#     model = load_model_structure(model_structure_path)
#     load_model_weights(model, model_weights_path)
#     return model

def complex_embedding(params):
    # datasets_dir, dataset_name, wordvec_initialization ='random', wordvec_path = None, loss = 'binary_crossentropy', optimizer = 'rmsprop', batch_size = 16, epochs= 4
    dataset_dir_path = os.path.join(params.datasets_dir, params.dataset_name)

    reader = data_reader_initialize(params.dataset_name,dataset_dir_path)

    lookup_table = None
    if not( (params.wordvec_initialization == 'random') | (params.wordvec_path == None)):
        if(params.wordvec_initialization == 'orthogonalize'):
            embedding_params = reader.get_word_embedding(params.wordvec_path,orthonormalized=True)
        else:
            embedding_params = reader.get_word_embedding(params.wordvec_path,orthonormalized=False)
        lookup_table = get_lookup_table(embedding_params)

    max_sequence_length = reader.max_sentence_length
    model = run_complex_embedding_network_2(lookup_table, max_sequence_length, reader.nb_classes)

    model.compile(loss = params.loss,
          optimizer = params.optimizer,
          metrics=['accuracy'])

    model.summary()

    train_test_val= reader.create_batch(embedding_params = embedding_params,batch_size = -1)

    training_data = train_test_val['train']
    test_data = train_test_val['test']
    validation_data = train_test_val['dev']


    # for x, y in batch_gen(training_data, max_sequence_length):
    #     model.train_on_batch(x,y)

    train_x, train_y = data_gen(training_data, max_sequence_length)
    test_x, test_y = data_gen(test_data, max_sequence_length)
    val_x, val_y = data_gen(validation_data, max_sequence_length)

    # assert len(train_x) == 67349
    # assert len(test_x) == 1821
    # assert len(val_x) == 872

    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
    val_y = to_categorical(val_y)

    history = model.fit(x=train_x, y = train_y, batch_size = params.batch_size, epochs= params.epochs,validation_data= (val_x, val_y))


    val_acc= history.history['val_acc']
    train_acc = history.history['acc']


    learning_curve_path = os.path.join(params.eval_dir,'learning_curve')
    line_1, = plt.plot(val_acc)
    line_2, = plt.plot(train_acc)
    # plt.axis([0, 6, 0, 20])

    plt.legend([line_1, line_2], ['val_acc', 'train_acc'])
    fig = plt.gcf()
    fig.savefig(learning_curve_path, dpi=fig.dpi)

    evaluation = model.evaluate(x = test_x, y = test_y)
    eval_file_path = os.path.join(params.eval_dir,'eval.txt')
    with open(eval_file_path,'w') as eval_file:
        eval_file.write('acc: {}, loss: {}'.format(evaluation[1], evaluation[0]))



if __name__ == '__main__':
    params = Params()
    params.parseArgs()
    complex_embedding(params)



    # dir_name = './data/CR'
    # reader = CRDataReader(dir_name)

    # dir_name = './data/MR'
    # reader = MRDataReader(dir_name)

    # dir_name = './data/MPQA'
    # reader = MPQADataReader(dir_name)


    # dir_name = './data/TREC'
    # reader = TRECDataReader(dir_name)

    # dir_name = './data/SUBJ'
    # reader = SUBJDataReader(dir_name)


    # dir_name = './data/SST'
    # reader = SSTDataReader(dir_name, nclasses = 5)


    # path_to_vec = 'glove/glove.6B.100d.txt'#





    # embedding_params = reader.get_word_embedding(path_to_vec,orthonormalized=False)
    # lookup_table = get_lookup_table(embedding_params)
    # # max_sequence_length = 10

    # max_sequence_length = reader.max_sentence_length
    # print(max_sequence_length)

    # model = run_complex_embedding_network_2(lookup_table, max_sequence_length, reader.nb_classes)
    # # model = run_real_network(lookup_table, max_sequence_length)
    # model.summary()

    # #################################################################
    # # # Training

    # # -1 refers to loading the whole data at once instead of in mini-batches
    # train_test_val= reader.create_batch(embedding_params = embedding_params,batch_size = -1)

    # training_data = train_test_val['train']
    # test_data = train_test_val['test']
    # validation_data = train_test_val['dev']


    # # for x, y in batch_gen(training_data, max_sequence_length):
    # #     model.train_on_batch(x,y)

    # train_x, train_y = data_gen(training_data, max_sequence_length)
    # test_x, test_y = data_gen(test_data, max_sequence_length)
    # val_x, val_y = data_gen(validation_data, max_sequence_length)

    # # assert len(train_x) == 67349
    # # assert len(test_x) == 1821
    # # assert len(val_x) == 872

    # train_y = to_categorical(train_y)
    # test_y = to_categorical(test_y)
    # val_y = to_categorical(val_y)
    # # print(y_binary)
    # history = model.fit(x=train_x, y = train_y, batch_size = 16, epochs= 4,validation_data= (val_x, val_y))


    # val_acc= history.history['val_acc']
    # train_acc = history.history['acc']


    # line_1, = plt.plot(val_acc)
    # line_2, = plt.plot(train_acc)
    # # plt.axis([0, 6, 0, 20])

    # plt.legend([line_1, line_2], ['val_acc', 'train_acc'])
    # fig = plt.gcf()
    # fig.savefig('learning_curve', dpi=fig.dpi)


    # model.save('my_model.h5')

    # model = load_model('my_model.h5', custom_objects = {'ComplexMultiply':ComplexMultiply})

    # model_json = model.to_json()
    # with open("model.json", "w") as json_file:
    #     json_file.write(model_json)


    # json_file = open('model.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()

    # loaded_model = model_from_json(loaded_model_json, custom_objects = {'ComplexMultiply':ComplexMultiply})



    # y = model.predict(x = test_x)
    # print(y)



    # save_model_structure(model, 'model/model_1')
    # save_model_weights(model, 'model/weight_1')

    # model = load_model('model/model_1','model/weight_1')


    #################################################################




