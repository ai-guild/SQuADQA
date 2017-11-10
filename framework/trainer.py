import sys
import tensorflow as tf
import numpy as np
import os

import logging
import pickle

from tqdm import tqdm


'''
    Trainer
        - (model, [trainfeed], [testfeed], [lr], [verbose])

'''
class Trainer(object):

    def __init__(self, model, 
            trainfeed=None, testfeed=None, # optional data feeds
            lr=0.001, # learning rate,
            verbose=0,
            interpret=None,
            dump_cache=False
            ):

        # get current session
        self._sess = tf.get_default_session()
        self._model = model
        self._trainfeed = trainfeed
        self._testfeed  = testfeed
        # display or skip info (verbose)
        self._v = lambda x : tqdm(x) if verbose else x
        # maintain performance stats
        self._stats = { 'accuracy' : [0.] }
        # mode
        self.TRAIN = 0
        self.EVAL  = 1
        # cache to store evaluation logs
        self.cache = []
        # interpreter call
        self._interpret = interpret
        # FLAG : dump cache
        self.dump_cache = dump_cache

    '''
        step
          - (fetch_list, feed_dict)
          - returns (loss, accuracy)
          - one forward(,[backward]) graph execution

    '''
    def step(self, fetch, feed_dict):
        results = self._sess.run(fetch, feed_dict=feed_dict)
        loss, accuracy = results[-2], results[-1]
        return loss, accuracy

    def execute(self, fetch, feed_dict):
        return self._sess.run(fetch, feed_dict=feed_dict)

    '''
        buildfeed
            - (datafeed, mode)
            - returns (feed_dict)
            - matches placeholders with batch data items
            - build feed_dict based on order

    '''
    def buildfeed(self, feed, mode, indices=False):
        ph = self._model.placeholders()
        batch = feed.next_batch()
        feed_dict = { ph[name] : batch[name]
                for name in ph.keys() }
        #feed_dict = { i:j for i,j in zip(ph, batch) }
        # add mode flag to feed dict
        feed_dict[self._model._mode] = mode

        if indices:
            return feed_dict, batch['idx']

        return feed_dict

    '''
        training module
            - (trainfeed, num_epochs)
            - returns (training stats)
            - handles batching, feed/fetch internally

    '''
    def train(self, feed=None,
            epochs=10000):

        feed = feed if feed else self._trainfeed

        # calc num of iterations per epoch
        iterations = feed.get_n()//feed.get_batch_size()
        
        fetch = [ self._model._train_op,
                self._model._loss,
                self._model._accuracy ]

        for i in range(epochs):
            # train in batches
            results = [ self.step(fetch, self.buildfeed(feed, self.TRAIN)) 
                    for j in self._v(range(iterations)) ]
            # evaluation after every epoch
            #  clear cache
            self.cache = []

            loss, accuracies = self.evaluate()
            
            print(':: [{}] loss : {}, accuracy : {}'.format(
                i, loss, accuracies))

            # select accuracy creterion
            accuracy = accuracies[0]

            # F1, Precision, Recall
            #self.alt_eval()

            # interpreter call
            if self._interpret:
                self._interpret(self.cache)

            # check for max accuracy
            if accuracy > max(self._stats['accuracy']):
                # get model parameters
                self._model_params = self._model.get_params()

            # stopping condition
            #  acc[i] < acc[i-1]
            if accuracy < self._stats['accuracy'][-1]:
                break

            # add current accuracy to stats
            self._stats['accuracy'].append(accuracy)

        # set best performing model params
        print(':: setting best model parameters')
        self._model.set_params(self._model_params)

        # run evaluate again, with best model parameters
        self.cache = []
        loss, accuracies = self.evaluate()
        print(':: [best model] Accuracy : ', accuracies)
        # dump cache to file
        if self.dump_cache:
            self.save_cache()

        # return stats
        return self._stats

    '''
        Evaluation
            - (testfeed)
            - returns (accuracy_mean)

    '''
    def evaluate(self, feed=None):
        ####
        # TODO : 
        #  add additional evaluation metrics from 
        #   evaluation.py
        #    (Precision, Recall, F1)
        feed = feed if feed else self._testfeed

        # calc num of iterations
        iterations = feed.get_n()//feed.get_batch_size()
        
        fetch = [ self._model._loss, self._model._accuracy, 
                self._model._probs, self._model._prediction ]

        # evaluate in batches
        losses, accuracies = [], []
        for j in range(iterations):
            # get feed dict
            feed_dict, indices = self.buildfeed(feed, self.EVAL, indices=True)
            loss, accuracy, probs, prediction = self.execute(fetch, feed_dict)
            losses.append(loss)
            accuracies.append(accuracy)
            # cache input, output - regardless
            #  ~~~if interpreter is enabled~~
            self.cache.append({ 'loss' : loss, 'accuracy' : accuracy,
                'prediction' : prediction, 'probs' : probs, 
                'idx' : indices })

        return np.array(losses).mean(axis=0), np.array(accuracies).mean(axis=0)

    '''
        F1, Precision, Recall

    '''
    def alt_eval(self, feed=None):
        feed = feed if feed else self._testfeed

        # calc num of iterations
        iterations = feed.get_n()//feed.get_batch_size()
        
        fetch = [ self._model._entity_prediction, self._model._ade_prediction ]

        ef1s, eps, ers = [], [], []
        af1s, aps, ars = [], [], []

        
        for j in range(iterations):
            feed_dict = self.buildfeed(feed, self.EVAL)
            eprediction, aprediction = self.execute(fetch, feed_dict)

            try:
                ground_truth = feed_dict[self._model._labels]
                seqlen = ground_truth.shape[-1]
                f1, precision, recall, _ = entity_f1_score(np.reshape(ground_truth, [seqlen,]),
                                                    np.reshape(eprediction, [seqlen,]))
                ef1s.append(f1)
                eps.append(precision)
                ers.append(recall)
                elog.setLevel(logging.INFO)
            except:
                elog.setLevel(logging.INFO)
                try:
                    
                    f1, precision, recall, _ = entity_f1_score(np.reshape(ground_truth, [seqlen,]),
                                                    np.reshape(eprediction, [seqlen,]))
                    elog.setLevel(logging.INFO)
                except:
                    print('==========')
                    print('incorrect targets: {} in entities'.format(ground_truth))
                    print(sys.exc_info())
                    elog.setLevel(logging.INFO)
            try:
                ground_truth = feed_dict[self._model._ade_labels]
                seqlen = ground_truth.shape[-1]
                f1, precision, recall, _ = f1_score(np.reshape(ground_truth, [seqlen,]),
                                                    np.reshape(aprediction, [seqlen,]))
                af1s.append(f1)
                aps.append(precision)
                ars.append(recall)
                elog.setLevel(logging.INFO)
            except:
                elog.setLevel(logging.DEBUG)
                try:
                    
                    f1, precision, recall, _ = f1_score(np.reshape(ground_truth, [seqlen,]),
                                                    np.reshape(aprediction, [seqlen,]))
                    elog.setLevel(logging.INFO)
                except:
                    print('==========')
                    print('incorrect targets: {} in ade'.format(ground_truth))
                    print(sys.exc_info())
                    elog.setLevel(logging.INFO)
            
                    
        print('entity F1 score:{}'.format(sum(ef1s)/len(ef1s)))
        print('entity precision:{}'.format(sum(eps)/len(eps)))
        print('entity recall:{}'.format(sum(ers)/len(ers)))
        
        print('ade F1 score:{}'.format(sum(af1s)/len(af1s)))
        print('ade precision:{}'.format(sum(aps)/len(aps)))
        print('ade recall:{}'.format(sum(ars)/len(ars)))


    def save_cache(self):
        if not os.path.exists('.cache'):
            os.makedirs('.cache')

        with open('.cache/train.cache', 'wb') as f:
            pickle.dump(self.cache, f)
