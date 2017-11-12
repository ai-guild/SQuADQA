import tensorflow as tf
import numpy as np

import sys
sys.path.append('../../')

from net.tensorflow.network import NeuralNetwork

DropoutWrapper = tf.nn.rnn_cell.DropoutWrapper
GRUCell = tf.nn.rnn_cell.GRUCell
MultiRNNCell= tf.nn.rnn_cell.MultiRNNCell


class AttentionSumReader(NeuralNetwork):

    def __init__(self, hdim, emb_dim, embedding, vocab_size,
            num_layers=1, dropout_value=0.5, lr=0.001):

        # build graph
        tf.reset_default_graph()

        # placeholders
        self._context = tf.placeholder(tf.int32, [None, None ], 
                name = 'context')
        self._query = tf.placeholder(tf.int32, [None, None], 
                name= 'query')
        self._answer = tf.placeholder(tf.int32, [None, ], 
                name= 'answer')

        # default placeholders
        self._mode = tf.placeholder(tf.int32, (), name='mode')

        # get dropout
        dropout = tf.cond(tf.equal(self._mode, 0),
                lambda : dropout_value, lambda : 0.)

        # infer dimensions [batch size, sequence length]
        batch_size_, max_context_len = tf.unstack(
                tf.shape( self._context))
        max_query_len = tf.shape(self._query)[1]
        context_lens = tf.count_nonzero(self._context, axis=1)
        query_lens  = tf.count_nonzero(self._query,   axis=1)

        # mask to compensate for padding <pad>
        padmask = tf.cast(self._context > 0, tf.float32)

        #  Word Embedding
        #   - initialized with Glove
        wemb = tf.get_variable(shape=[vocab_size, emb_dim], dtype=tf.float32,
                        initializer=tf.constant_initializer(embedding),
                        #initializer=tf.random_uniform_initializer(-0.01, 0.01),
                           name='word_embedding')

        # account for padding and unknown
        wemb = tf.concat([ tf.zeros([2, emb_dim]), wemb ], axis=0)

        # Encoder
        #  o Context Encoder

        # define Multi-GRU cell with dropout
        cell = lambda : MultiRNNCell([ DropoutWrapper(GRUCell(hdim), 
            output_keep_prob=1. - dropout) for _ in range(num_layers) ])

        with tf.variable_scope('context_encoder') as scope:
            # Bidirectional GRU network
            #  encodes embedded context [B, T, emb_dim] -> [B, T, hdim*2]
            states, _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw = cell(), cell_bw = cell(),
                    inputs = tf.nn.embedding_lookup(wemb, 
                        self._context), # embed context 
                    swap_memory=True, 
                    sequence_length=context_lens, # lens without padding 
                    dtype=tf.float32)

            # Context Representation : combine forward, backward states
            context = tf.squeeze(tf.concat(states, axis=-1))

        #
        #  o Query Encoder
        #
        with tf.variable_scope('query_encoder') as scope:
            # Bidirectional GRU network
            #  encodes embedded query to fixed length vector
            #   [B, T, emb_dim] -> [B, hdim*2]
            _, final_state = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw= cell(), cell_bw= cell(),
                    inputs = tf.nn.embedding_lookup(wemb, 
                        self._query), # embed context 
                    swap_memory=True, 
                    sequence_length=query_lens, # lens without padding 
                    dtype=tf.float32)

            # Query Representation : final state of encoder
            #  combine the forward and backward state 
            #   (last layer of stacking)
            query = tf.concat( [ final_state[0][-1], final_state[1][-1] ], 
                    axis=-1)

        with tf.variable_scope('attention') as scope:
            # calculate attention as dot-product 
            #  between context states and query
            attention = tf.squeeze(tf.matmul(
                tf.expand_dims(query, axis=-1), 
                context, adjoint_a=True, adjoint_b=True))

        # probabilities
        self._probs = tf.nn.softmax(attention*padmask)

        # prediction
        self._prediction = tf.argmax(self._probs, axis=-1)

        # cross entropy
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=attention * padmask, 
                labels=self._answer)

        # loss
        self._loss = tf.reduce_mean(ce)

        # calculate and group accuracy
        self._accuracy = self.calc_accuracy(self._answer, self._probs)

        # optimization 
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self._train_op = optimizer.minimize(self._loss)

    def placeholders(self):
        return {
                'context'  : self._context,
                'question' : self._query,
                'answer'   : self._answer
                }

    def calc_accuracy(self, labels, probs):
        # calculate accuracy
        correct_labels = tf.equal(
                tf.cast(labels, tf.int64), 
                tf.argmax(probs, axis=-1))

        return tf.reduce_mean(tf.cast(correct_labels, 
            tf.float32))


def execute_graph(model, t):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        results = sess.run(t, feed_dict = {
                model._context : np.random.randint(1, 100, [8, 20]),
                model._query   : np.random.randint(1, 100, [8, 10]),
                model._answer  : np.random.randint(0, 20,  [8, ]),
                model._mode    : 0
                })

        return results

    
if __name__ == '__main__':
    model = AttentionSumReader(hdim=32, emb_dim=16,
            embedding=np.random.uniform(-0.01, 0.01, [100, 16]),
            vocab_size=100, num_layers=1)

    results = execute_graph(model, model._accuracy)
    print(results)
