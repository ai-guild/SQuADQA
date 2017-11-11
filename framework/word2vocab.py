import os
import pickle
import numpy as np

from word2vec import Word2Vec

VOCABFILE = '.cache/vocabulary.txt'
EMBFILE   = '.cache/embeddings.{}d.bin'


class Word2Vocab(object):

    def __init__(self, dim):
        # read model from bin
        print(':: loading vocabulary from disk')
        self.dim = dim

        # get list of words from vocab
        #  get subset of word2vec model
        #   account for padding and unknown
        self._vocab = ['PAD', 'UNK'] + sorted(list(
            set(open(VOCABFILE).read().splitlines())))
        
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

    def load_embeddings(self):

        # check if embeddings saved in cache
        if os.path.exists(EMBFILE.format(self.dim)):
            # read from cache; return
            return pickle.load(open(
                EMBFILE.format(self.dim), 'rb'))

        # read model from word2vec
        model = Word2Vec(self.dim).get_model()
        embeddings = []
        for w in self._vocab:
            # if word in model
            if w in model:
                emb = model[w]
            # else check if lower-case of w in model
            elif w.lower() in model:
                emb =  model[w.lower()]
            # return zero vector
            else:
                emb = np.zeros(self.dim)
            # keep track of embedding 
            embeddings.append(emb)
        # np.array
        embeddings = np.stack(embeddings)
        # attach to self 
        self.emb = embeddings
        # write to cache
        pickle.dump(self.emb, open(
            EMBFILE.format(self.dim), 'wb'))
        # make sure vocab size == num of embeddings
        assert self.vocab_size() == self.emb.shape[0]

        return self.emb


if __name__ == '__main__':

    w = Word2Vocab(50)
    w.load_embeddings()
