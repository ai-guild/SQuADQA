import os
import pickle
import random

import gensim
import numpy as np

PUBMED_FILEPATH = '../datasets/w2v/PubMed-w2v.bin'
VOCAB_FILEPATH = 'vocabulary.txt'

class Word2Vocab(object):

    def __init__(self, filepath=VOCAB_FILEPATH, d=200):
        # read model from bin
        print(':: loading vocabulary from disk')
        self.d = d

        # get list of words from vocab
        #  get subset of word2vec model
        #   account for padding and unknown
        self._vocab = ['PAD', 'UNK'] + sorted(list(set(open(filepath).read().splitlines())))
        
        self._lookup = {w:i for i, w in enumerate(self._vocab)}
        
    def encode(self, w):

        if type(w) == type([]):
            return [ self.encode(wi) for wi in w ]

        if ' ' in w:
            return [ self.encode(wi) for wi in w.split(' ') ]

        if w in self._lookup:
            return self._lookup[w]

        if w.lower() in self._lookup:
            return self._lookup[w.lower()]

        return 0
        
    def zero(self):
        return 0

    def lookup(self, idx):
        # check if integer
        if type(idx) == type(21):
            return self._vocab[idx]
        # check if list of ints
        if type(idx) == type([]):
            return [ self._vocab[i] for i in idx ]

    def fetch_lookup(self):
        return self._lookup[2:]

    def vocab_size(self):
        return len(self._vocab)

    def load_embeddings(self, pretrained_model=PUBMED_FILEPATH, filepath='embeddings.bin'):
        if os.path.exists(filepath):
            self.emb = pickle.load(open(filepath, 'rb'))

        else:
        
            model = gensim.models.KeyedVectors.load_word2vec_format(pretrained_model,
                                                                    binary=True)
            embeddings = []
            for w in self._vocab:
                if w in model:
                    emb = model[w]
                    
                elif w.lower() in model:
                    emb =  model[w.lower()]

                else:
                    emb = np.zeros(self.d)
              
                embeddings.append(emb)

            embeddings = np.stack(embeddings)
            
            self.emb = embeddings
            pickle.dump(self.emb, open(filepath, 'wb'))

        assert self.vocab_size() == self.emb.shape[0]

        return self.emb
                    
        
if __name__ == '__main__':

    w = Word2Vocab()
    w.load_embeddings()
