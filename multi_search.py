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
nb_classes=2
def createModel(dropout_rate=0.5,optimizer='adam',learning_rate=0.1,init_criterion="he",projection= True,activation="relu"):
#    projection= True,max_sequence_length=56,nb_classes=2,dropout_rate=0.5,embedding_trainable=True,random_init=False

        

    
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
    from keras import optimizers
    if optimizer=="Nadam":
        optimizer=optimizers.Nadam(lr=learning_rate,clipvalue=0.5)
    else:
        optimizer=optimizers.Adam(lr=learning_rate,clipvalue=0.5)
#    optimizer = optimizers.SGD(lr=learning_rate, momentum=momentum)
#    optimizer=Adadelta(lr=1.0, rho=0.95, epsilon=1e-09) 
    model = Model(sequence_input, output)
    model.compile(loss ="binary_crossentropy",
          optimizer = optimizer,
          metrics=['accuracy'])
    
    return model


params = Params()
params.parse_config('config/waby.ini')

import argparse
parser = argparse.ArgumentParser(description='running the complex embedding network')
parser.add_argument('-gpu', action = 'store', dest = 'gpu', help = 'please enter the gpu num.')
parser.add_argument('-count', action = 'store', dest = 'count', help = 'count.')
parser.add_argument('-dataset', action = 'store', dest = 'dataset', help = 'please enter the dataset.')
args = parser.parse_args()
try:
    gpu = int(args.gpu)
except:
    gpu=0
try:
    count = int(args.count)
except:
    count=8
try :
    if args.dataset is not None:
        params.dataset_name = args.dataset
except:
    pass     
print("gpu : %d" % gpu)
print("dataset: " + params.dataset_name)

reader = data_reader_initialize(params.dataset_name,params.datasets_dir)
nb_classes=reader.nb_classes

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

train_y = to_categorical(train_y)
test_y = to_categorical(test_y)
val_y = to_categorical(val_y)

def run_task(zipped_args):
    i,(dropout_rate,optimizer,learning_rate,init_mode,projection,batch_size,activation) = zipped_args

    arg_str=(" ".join([str(ii) for ii in (dropout_rate,optimizer,learning_rate,init_mode,projection,batch_size,activation)]))
    print ('Run task %s (%d)... \n' % (arg_str, os.getpid()))
#    try:
#        GPUUtil.setCUDA_VISIBLE_DEVICES(num_GPUs=1, verbose=True) != 0
#    except Exception as e:
#        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(i%8))
#        print ('use GPU %d \n' % (int(i%8)))
#    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(i%8))
#    print ('use GPU %d \n' % (int(i%8)))    
    model = createModel(dropout_rate,optimizer,learning_rate,init_mode,activation)
    print(dropout_rate,optimizer,learning_rate,init_mode,activation)

    start=time.time()
    history = model.fit(x=train_x, y = train_y, batch_size = batch_size, epochs= params.epochs,validation_data= (test_x, test_y),verbose = 0 )
    
    val_acc= history.history['val_acc']
    train_acc = history.history['acc']
    
    model_info = "%s:  dropout:%.2f  opti:%s init: %s batch_size:%d  activation:%s lr:%f" %("mixture" if projection else "superposition",dropout_rate,optimizer,init_mode,batch_size,activation,learning_rate )    
   
    df = pd.read_csv(params.dataset_name+".csv",index_col=0,sep="\t")
    dataset = params.dataset_name
#    if arg_str not in df:
#        df.loc[arg_str] = pd.Series()
#    if dataset not in df.loc[arg_str]:
    df.loc[model_info,dataset] = max(val_acc) 
    df.to_csv(params.dataset_name+".csv",sep="\t")
    
    print(model_info +" with time :"+ str( time.time()-start)+" ->" +str( max(val_acc) ) )

        




#    time.sleep(1)
if __name__ == "__main__":

    if not os.path.exists(params.dataset_name+".csv"):
        with open(params.dataset_name+".csv","w") as f:
            f.write("argument\t"+params.dataset_name+"\n")
#            f.write("0\n")
            f.close()

    
#    dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    dropout_rates = [0.0, 0.1, 0.2,  0.5]  
#    optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    optimizers = [ 'Adam', 'Nadam']
#    init_modes = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform','he']
    learning_rates=[10,1,1e-1,1e-2,1e-3]
    init_modes = ["glorot","he"]
    projections=  [True,False]
    batch_sizes = [8,32,64,128]
    activations=["relu","sigmoid","tanh"]
    parameter_pools=[
            ("dropout_rates",[0.0, 0.1, 0.2]),
            ("optimizers",[ 'Adam', 'Nadam']),
            ("learning_rates",[10,1,1e-1,1e-2,1e-3]),
            ("init_modes",["glorot","he"]),
            ("projections",[True,False]),
            ("batch_sizes",[8,32,64,128]),
            ("activations",["relu","sigmoid","tanh"])
            ]
    pool =[ arg for arg in itertools.product(*[paras[1] for paras in parameter_pools] )]
    random.shuffle(pool)
    args=[(i,arg) for i,arg in enumerate(pool) if i%count==gpu]    

#    args=[i for i in enumerate(itertools.product(dropout_rates,optimizers,learning_rates,init_modes,projections,batch_sizes,activations)) if i[0]%8==gpu]

    for arg in args:

        run_task(arg)


    









