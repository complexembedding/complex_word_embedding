# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 21:44:09 2018

@author: wabywang
"""


import sys
import os,time,random
import numpy as np
import codecs
import pandas as pd
sys.path.append('complexnn')
from keras.models import Model, Input, model_from_json, load_model
from keras.layers import Embedding, GlobalAveragePooling1D,Dense, Masking, Flatten, Dropout
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




import itertools
import multiprocessing
import GPUUtil

def createModel(dropout_rate=0.5,optimizer='adam',init_criterion="he",projection= True,):
#    projection= True,max_sequence_length=56,nb_classes=2,dropout_rate=0.5,embedding_trainable=True,random_init=False

        
    max_sequence_length=56
    nb_classes=2
    embedding_trainable=True
    # can be searched by grid
    random_init=False


    embedding_dimension = lookup_table.shape[1]
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')

    phase_embedding =Dropout(dropout_rate) (phase_embedding_layer(max_sequence_length, lookup_table.shape[0], embedding_dimension, trainable = embedding_trainable)(sequence_input))

    
    amplitude_embedding = Dropout(dropout_rate)(amplitude_embedding_layer(np.transpose(lookup_table), max_sequence_length, trainable = embedding_trainable, random_init = random_init)(sequence_input))
    
    
    [seq_embedding_real, seq_embedding_imag] = ComplexMultiply()([phase_embedding, amplitude_embedding])

    if(projection):
        [sentence_embedding_real, sentence_embedding_imag]= ComplexMixture()([seq_embedding_real, seq_embedding_imag])
        sentence_embedding_real = Flatten()(sentence_embedding_real)
        sentence_embedding_imag = Flatten()(sentence_embedding_imag)    
        
    else:
        [sentence_embedding_real, sentence_embedding_imag]= ComplexSuperposition()([seq_embedding_real, seq_embedding_imag])

    # output = Complex1DProjection(dimension = embedding_dimension)([sentence_embedding_real, sentence_embedding_imag])
    predictions = ComplexDense(units = nb_classes,init_criterion=init_criterion, activation='sigmoid', bias_initializer=Constant(value=-1))([sentence_embedding_real, sentence_embedding_imag])

    output = GetReal()(predictions)
    model = Model(sequence_input, output)
    model.compile(loss ="binary_crossentropy",
          optimizer = optimizer,
          metrics=['accuracy'])
    
    return model


params = Params()
params.parse_config('config/waby.ini')

reader = data_reader_initialize(params.dataset_name,params.datasets_dir)

if(params.wordvec_initialization == 'orthogonalize'):
    embedding_params = reader.get_word_embedding(params.wordvec_path,orthonormalized=True)

elif( (params.wordvec_initialization == 'random') | (params.wordvec_initialization == 'word2vec')):
    embedding_params = reader.get_word_embedding(params.wordvec_path,orthonormalized=False)
else:
    raise ValueError('The input word initialization approach is invalid!')

# print(embedding_params['word2id'])
lookup_table = get_lookup_table(embedding_params)

max_sequence_length = reader.max_sentence_length
random_init = True
if not(params.wordvec_initialization == 'random'):
    random_init = False

train_test_val= reader.create_batch(embedding_params = embedding_params,batch_size = -1)

training_data = train_test_val['train']
test_data = train_test_val['test']
validation_data = train_test_val['dev']

train_x, train_y = data_gen(training_data, max_sequence_length)
test_x, test_y = data_gen(test_data, max_sequence_length)
val_x, val_y = data_gen(validation_data, max_sequence_length)


def run_task(zipped_args):
    i,(dropout_rate,optimizer,init_mode,projection) = zipped_args

    arg_str=(" ".join([str(ii) for ii in (dropout_rate,optimizer,init_mode,projection)]))
    print ('Run task %s (%d)... \n' % (arg_str, os.getpid()))
#    try:
#        GPUUtil.setCUDA_VISIBLE_DEVICES(num_GPUs=1, verbose=True) != 0
#    except Exception as e:
#        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(i%8))
#        print ('use GPU %d \n' % (int(i%8)))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(i%8))
    print ('use GPU %d \n' % (int(i%8)))    
    model = createModel(dropout_rate,optimizer,init_mode)
    print("begin to train")
    history = model.fit(x=train_x, y = train_y, batch_size = 1, epochs= params.epochs,validation_data= (test_x, test_y))

    val_acc= history.history['val_acc']
    train_acc = history.history['acc']
    with open("eval.txt") as f:
        model_info = "%.4f test acc,  %4.f train acc , model : %s,  dropout_rate: %.2f, optimizer: %s ,init_mode %s \n " %(max(val_acc),max(train_acc),"mixture" if projection else "superposition",dropout_rate,optimizer,init_mode )    
        f.write(model_info)
    
      



#    time.sleep(1)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='running the complex embedding network')
    parser.add_argument('-gpu_num', action = 'store', dest = 'gpu', help = 'please enter the gpu num.')
    args = parser.parse_args()
    gpu = int(args["gpu"])
    dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] 
    optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    init_modes = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform','he']
    projections=  [True,False]
    

    args=[i for i in itertools.product(dropout_rates,optimizers,init_modes,projections) if i[0]==gpu]

    for arg in enumerate(args):
        run_task(arg)


    









