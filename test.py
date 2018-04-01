# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 22:14:39 2018

@author: wabywang
"""

import time,random,os,itertools
#def long_time_task(zipped_args):
#    i,dropout_rate,optimizer,init_mode = zipped_args
#    os.environ["CUDA_VISIBLE_DEVICES"] = str(i)
#    print ('Run task %s (%s)... with GPU i% \n' % (" ".join((dropout_rate,optimizer,init_mode)), os.getpid()), i)
#    start = time.time()
#    time.sleep(random.random() * 3)
#    end = time.time()
#    print ('Task %s runs %0.2f seconds.' % (args, (end - start)))
#
#
#dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] 
#optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
#init_modes = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform','he']
#args=[i for i in itertools.product(dropout_rates,optimizers,init_modes)]
#from multiprocessing import Process, Pool
#
#if __name__ == "__main__":
#    p = Pool(8)
#    p.apply_async(long_time_task, args=(args,))
#
#    print ('Waiting for all subprocesses done...')
#    p.close()
#    p.join()
#    print ('All subprocesses done.')


import multiprocessing
import time
def long_time_task(zipped_args):
    i,(dropout_rate,optimizer,init_mode) = zipped_args
    print(zipped_args)
    arg_str=(" ".join([str(ii) for ii in (dropout_rate,optimizer,init_mode)]))
    print(arg_str)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(i%8))
    print ('Run task %s (%d)... with GPU %d \n' % (arg_str, os.getpid(), int(i%8)))
#    time.sleep(1)
if __name__ == "__main__":
    dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] 
    optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    init_modes = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform','he']
    args=[i for i in itertools.product(dropout_rates,optimizers,init_modes)]
    pool = multiprocessing.Pool(processes=4)
    for arg in enumerate(args):
        pool.apply_async(long_time_task, (arg, ))
    pool.close()
    pool.join()
    print ("Sub-process(es) done.")