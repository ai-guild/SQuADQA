import numpy as np
from info import *

FILEPATH='../dataset/glove/glove.6B.{}d.txt'
dims = [ 50 , 100, 200, 300 ]


class Word2Vec(object):

    def __init__(self, dim, vocab=None, binfile=FILEPATH):
        # dimensions
        assert dim in dims
        self.dim = dim

        # read model from file
        #self.model = gensim.models.KeyedVectors.load_word2vec_format(binfile,
        #        binary=True)
        self.model = self.read()

        # get list of words from vocab
        #  get subset of word2vec model
        #   account for padding and unknown
        if vocab:
            self._vocab = vocab
            self._lookup = np.concatenate( [np.zeros([2, self.dim]), 
                    self.encode(self._vocab)], axis=0 )
        #        self.encode(self._vocab.as_list())], axis=0 )
        # add zeros

    '''
        Read from disk

    '''
    def read(self):
        print(':: loading Glove model from disk')
        f = open(FILEPATH.format(self.dim),'r')
        model = {}
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
        Ip('{} words loaded!'.format(len(model)))

        return model
        
    def encode(self, w):
        # check if input is list
        if type(w) == type([]):
            return np.array([ self.encode(wi) 
                for wi in w ])
        # check if input is sentence
        if ' ' in w:
            return np.array([ self.encode(wi)
                for wi in word_tokenize(w) ])
        # check if word exists in model
        if w in self.model:
            return self.model[w]

        if w.lower() in self.model:
            return self.model[w.lower()]

        # check if hyphenate
        #if '-' in w or ':' in w:
        #    vecs = [ self.encode(wi) 
        #            for wi in w.split('-') ]
        #    # return mean of words in hyphenate
        #    return np.array(vecs).mean(axis=0)

        # unknown word
        return self.zero()

    def zero(self):
        return np.zeros(self.dim)

    def lookup(self, idx):
        # check if integer
        if type(idx) == type(21):
            return self._lookup[idx]
        # check if list of ints
        if type(idx) == type([]):
            return [ self._lookup[i] for i in idx ]

    def fetch_lookup(self):
        return self._lookup[2:]

    def get_model(self):
        return self.model
