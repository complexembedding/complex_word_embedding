from __future__ import absolute_import, division, unicode_literals

import io
import numpy as np
import logging
import cmath
import random
import math

word_list = []

# Create dictionary
def create_dictionary(sentences, threshold=0):
    words = {}
    for s in sentences:
        for word in s:
            words[word] = words.get(word, 0) + 1

    if threshold > 0:
        newwords = {}
        for word in words:
            if words[word] >= threshold:
                newwords[word] = words[word]
        words = newwords
    words['<s>'] = 1e9 + 4
    words['</s>'] = 1e9 + 3
    words['<p>'] = 1e9 + 2

    sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i

    return id2word, word2id




def form_matrix(file_name):
    ll = []
    f = open(file_name, 'r')
    lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i].split()
        word_list.append(line[0])
        a = line[1:]
        ll.append(a)
    matrix = np.asarray(ll)
    return matrix



def orthonormalized_word_embeddings(word_embeddings_file):
    matrix = form_matrix(word_embeddings_file)
    print 'Initial matrix constructed!'
    matrix_norm = np.zeros((matrix.shape[0], matrix.shape[1]))
    matrix = matrix.astype(np.float)
    matrix_sum = np.sqrt(np.sum(np.square(matrix), axis=1))
    for i in range(np.shape(matrix)[0]):
        matrix_norm[i] = matrix[i]/matrix_sum[i]
    print 'Matrix normalized'

    ##q - basis vectors(num_words x dimension). 
    ##r - coefficients of each word in the basis(dimension x num_words)
    q, r = np.linalg.qr(np.transpose(matrix_norm), mode = 'complete') 
    print 'qr factorization completed. Matrix orthogonalized!'

    ## Dot product of king and prince vectors same as in the original embeddings (0.76823)
    king = word_list.index('king')
    prince = word_list.index('prince')
    print (np.dot(r[:, king], r[:, prince]))
    return r


# Get word vectors from vocabulary (glove, word2vec, fasttext ..)
def get_wordvec(path_to_vec, word2id=None):
    coefficients_matrix = orthonormalized_word_embeddings(path_to_vec)
    word_vec = {}

    with io.open(path_to_vec, 'r', encoding='utf-8') as f:
        # if word2vec or fasttext file : skip first line "next(f)"
        for word in word_list:
            word, vec = line.split(' ', 1)
            if word in word2id:
                word_vec[word] = coefficients_matrix[:, word_list.index(word)]

    logging.info('Found {0} words with word vectors, out of \
        {1} words'.format(len(word_vec), len(word2id)))
    return word_vec

# Set word phases
# Currently only using random phase
def set_wordphase(word2id):
    word2phase = {}
    for word in word2id.keys():
        word2phase[word] = random.random()*2*math.pi

    return word2phase
def get_batch(embedding_params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []

    for sent in batch:
        sentvec = []
        for word in sent:
            if word in embedding_params['word_vec']:
                wordvec = embedding_params['word_vec'][word]
                if word in embedding_params['word_complex_phase']:
                    complex_phase = embedding_params['word_complex_phase'][word]
                    wordvec = [x * cmath.exp(1j*complex_phase) for x in wordvec]
                sentvec.append(wordvec)
        if not sentvec:
            vec = np.zeros(embedding_params['wvec_dim'])
            sentvec.append(vec)
        # sentvec = np.mean(sentvec, 0)
        embeddings.append(sentvec)

    embeddings = np.vstack(embeddings)
    return embeddings


def main():
    word_vec = get_wordvec('/mnt/c/Users/su632/Downloads/glove.6B/glove.6B.100d.txt') #path to word_vec file
    print len(word_vec) #should be size of vocab
    print word_vec['the']  #should be a vector with first element 1 and all other zeros

if __name__ == '__main__':
        main()
        
