import gensim
import numpy as np

FILEPATH='datasets/w2v/PubMed-w2v.bin'

class Word2Vec(object):

    def __init__(self, vocab=None, binfile=FILEPATH, d=200):
        # read model from bin
        print(':: loading word2vec from disk')
        self._model = gensim.models.KeyedVectors.load_word2vec_format(binfile,
                binary=True)

        # dimensions
        self.d = d

        # get list of words from vocab
        #  get subset of word2vec model
        #   account for padding and unknown
        if vocab:
            self._vocab = vocab
            self._lookup = np.concatenate( [np.zeros([2, self.d]), 
                    self.encode(self._vocab)], axis=0 )
        #        self.encode(self._vocab.as_list())], axis=0 )
        # add zeros
        
    def encode(self, w):
        # check if input is list
        if type(w) == type([]):
            return np.array([ self.encode(wi) 
                for wi in w ])
        # check if input is sentence
        if ' ' in w:
            return np.array([ self.encode(wi)
                for wi in w.split(' ') ])
        # check if word exists in model
        if w in self._model:
            return self._model[w]

        if w.lower() in self._model:
            return self._model[w.lower()]

        # check if hyphenate
        #if '-' in w or ':' in w:
        #    vecs = [ self.encode(wi) 
        #            for wi in w.split('-') ]
        #    # return mean of words in hyphenate
        #    return np.array(vecs).mean(axis=0)

        # unknown word
        return self.zero()

    def zero(self):
        return np.zeros(self.d)

    def lookup(self, idx):
        # check if integer
        if type(idx) == type(21):
            return self._lookup[idx]
        # check if list of ints
        if type(idx) == type([]):
            return [ self._lookup[i] for i in idx ]

    def fetch_lookup(self):
        return self._lookup[2:]
