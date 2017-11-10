import numpy as np

from word2vec import Word2Vec
#from random import shuffle


'''
    DataFeed
        - (list_of_datapoints)
        - [ datapoints ] -> batch_i 
        - [ datapoints ] -> next_batch

'''
class DataFeed(object):

    def __init__(self, batchop, batch_size=1, 
            datapoints = [], w2v=None):

        # current iteration
        self._offset = 0

        # num of examples
        self.n = len(datapoints)

        # default batch size
        self.B = batch_size

        # create Word2Vec model
        self._w2v = Word2Vec() if not w2v else w2v

        # batch process operation
        self._batchop = batchop

        # if data available
        if len(datapoints):
            self.bind(datapoints)

    '''
        bind data to feed

    '''
    def bind(self, datapoints):
        # start over
        self._offset = 0
        # get num of examples
        self._n = len(datapoints)
        # bind data to instance
        self._data = datapoints

    '''
        get num of examples

    '''
    def get_n(self):
        return self.n

    '''
        get batch size

    '''
    def get_batch_size(self):
        return self.B

    '''
        get batch next to offset

    '''
    def batch(self):

        batch_size = self.B

        # fetch batch next to offset
        s, e = self._offset, self._offset + batch_size

        # update offset
        self._offset += batch_size

        # fetch datapoints
        datapoints = self._data[s:e]

        # shuffle datapoints
        # shuffle(datapoints)

        # apply model-specific operation on batch
        feed_data = self._batchop(datapoints, self._w2v)

        # add indices
        feed_data['idx'] = [ dp._idx for dp in datapoints ]

        return feed_data

    '''
        get next batch

    '''
    def next_batch(self):

        batch_size = self.B

        def next_batch_():
            # check limit
            if self._offset + batch_size > self.n:
                # star over
                self._offset = 0
            # return batch
            return self.batch()

        return next_batch_()
