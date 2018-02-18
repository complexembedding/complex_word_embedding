from __future__ import absolute_import, division, unicode_literals

import io
import numpy as np
import logging
import cmath
import random
import math


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


# Get word vectors from vocabulary (glove, word2vec, fasttext ..)
def get_wordvec(path_to_vec, word2id):
    word_vec = {}

    with io.open(path_to_vec, 'r', encoding='utf-8') as f:
        # if word2vec or fasttext file : skip first line "next(f)"
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word2id:
                word_vec[word] = np.fromstring(vec, sep=' ')

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
