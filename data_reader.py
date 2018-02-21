import os
import io
import logging
import numpy as np
import data as data
# PATH_TO_GLOVE = 'glove/glove.6B.100d.txt'

class SSTDataReader(object):
    def __init__(self, task_dir_path, nclasses=2, seed=1111):
        self.seed = seed

        # binary of fine-grained
        assert nclasses in [2, 5]
        self.nclasses = nclasses
        self.task_name = 'Binary' if self.nclasses == 2 else 'Fine-Grained'
        logging.debug('***** Transfer task : SST %s classification *****\n\n', self.task_name)

        train = self.loadFile(os.path.join(task_dir_path, self.task_name,'sentiment-train'))
        dev = self.loadFile(os.path.join(task_dir_path, self.task_name, 'sentiment-dev'))
        test = self.loadFile(os.path.join(task_dir_path, self.task_name, 'sentiment-test'))
        self.sst_data = {'train': train, 'dev': dev, 'test': test}

    def get_word_embedding(self, path_to_vec):
        samples = self.sst_data['train']['X'] + self.sst_data['dev']['X'] + \
                self.sst_data['test']['X']

        id2word, word2id = data.create_dictionary(samples, threshold=0)
        word_vec = data.get_wordvec(path_to_vec, word2id)
        wvec_dim = len(word_vec[next(iter(word_vec))])

        #stores the value of theta for each word
        word_complex_phase = data.set_wordphase(word2id)

        params = {'word2id':word2id, 'word_vec':word_vec, 'wvec_dim':wvec_dim,'word_complex_phase':word_complex_phase,'id2word':id2word}

        return params

    def loadFile(self, fpath):
        sst_data = {'X': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                if self.nclasses == 2:
                    sample = line.strip().split('\t')
                    sst_data['y'].append(int(sample[1]))
                    sst_data['X'].append(sample[0].split())
                elif self.nclasses == 5:
                    sample = line.strip().split(' ', 1)
                    sst_data['y'].append(int(sample[0]))
                    sst_data['X'].append(sample[1].split())
        assert max(sst_data['y']) == self.nclasses - 1
        return sst_data

    def create_batch(self, embedding_params, batch_size = 1):
        sst_embed = {'train': {}, 'dev': {}, 'test': {}}
        for key in self.sst_data:
            sst_embed[key] = {'X':[],'y':[]}
            logging.info('Computing embedding for {0}'.format(key))
            sorted_data = sorted(zip(self.sst_data[key]['X'],
                                     self.sst_data[key]['y']),
                                 key=lambda z: (len(z[0]), z[1]))
            self.sst_data[key]['X'], self.sst_data[key]['y'] = map(list, zip(*sorted_data))

            for ii in range(0, len(self.sst_data[key]['y']), batch_size):
                batch = self.sst_data[key]['X'][ii:ii + batch_size]
                embeddings = data.get_index_batch(embedding_params, batch)
                # print(embeddings)
                sst_embed[key]['X'].append(embeddings)
                # print(self.sst_data[key]['y'][ii:ii + batch_size])
                sst_embed[key]['y'].append(self.sst_data[key]['y'][ii:ii + batch_size])
            # sst_embed[key]['X'] = np.vstack(sst_embed[key]['X'])
            # print(sst_embed[key]['y'])
            sst_embed[key]['y'] = np.array(sst_embed[key]['y'])
            # print(sst_embed[key]['y'])
            logging.info('Computed {0} embeddings'.format(key))
        return sst_embed


if __name__ == '__main__':
    dir_name = 'C:/Users/quartz/Documents/python/complex_word_embedding/'
    path_to_vec = 'glove/glove.6B.100d.txt'#
    reader = SSTDataReader(dir_name,nclasses = 2)
    params = reader.get_word_embedding(path_to_vec)
    # print(params['word_vec'])
    sentences = reader.create_batch(embedding_params = params,batch_size = 3)
    batches = sentences['train']['X']
    labels = sentences['train']['y']
    print(len(batches))
    print(len(labels))
