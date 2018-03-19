import os
import io
import logging
import numpy as np
import configparser
import argparse
class Params(object):
    def __init__(self, datasets_dir = None, dataset_name = None, wordvec_initialization ='random', wordvec_path = None, eval_dir = None, loss = 'binary_crossentropy', optimizer = 'rmsprop', batch_size = 16, epochs= 4):
        self.datasets_dir = datasets_dir
        self.dataset_name = dataset_name
        self.wordvec_initialization = wordvec_initialization
        self.wordvec_path = wordvec_path
        self.loss = loss
        self.optimizer = optimizer
        self.batch_size =batch_size
        self.epochs = epochs
        self.eval_dir = eval_dir

    def parse_config(self, config_file_path):
        config = configparser.ConfigParser()
        config.read(config_file_path)
        config_common = config['COMMON']
        self.datasets_dir = config_common['datasets_dir']
        self.dataset_name = config_common['dataset_name']
        self.wordvec_initialization = config_common['wordvec_initialization']
        self.wordvec_path = config_common['wordvec_path']
        self.loss = config_common['loss']
        self.optimizer = config_common['optimizer']
        self.batch_size = int(config_common['batch_size'])
        self.epochs = int(config_common['epochs'])
        self.eval_dir = config_common['eval_dir']

    def export_to_config(self, config_file_path):
        config = configparser.ConfigParser()
        config['COMMON'] = {}
        config_common = config['COMMON']
        config_common['datasets_dir'] = self.datasets_dir
        config_common['dataset_name'] = self.dataset_name
        config_common['wordvec_initialization'] = self.wordvec_initialization
        config_common['wordvec_path'] = self.wordvec_path
        config_common['loss'] = self.loss
        config_common['optimizer'] = self.optimizer
        config_common['batch_size'] = str(self.batch_size)
        config_common['epochs'] = str(self.epochs)
        config_common['eval_dir'] = str(self.eval_dir)
        with open(config_file_path, 'w') as configfile:
            config.write(configfile)

    def parseArgs(self):
        #required arguments:
        parser = argparse.ArgumentParser(description='running the complex embedding network')
        parser.add_argument('-config', action = 'store', dest = 'config_file_path', help = 'The configuration file path.')
        args = parser.parse_args()
        self.parse_config(args.config_file_path)


