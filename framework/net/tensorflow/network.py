import tensorflow as tf

'''
    Neural Network wrapper

'''
class NeuralNetwork(object):

    def __init__(self, *args, **kwargs):
        self._optimizer = tf.train.AdamOptimizer

    '''
        set model parameters

    '''
    def set_params(self, params):
        # get session
        sess = tf.get_default_session()
        # define operation
        op = [ tf.assign(var, val) for var, val in
                zip(tf.trainable_variables(), params) ]
        # run operation
        return sess.run(op)

    '''
        get model parameters

    '''
    def get_params(self):
        # get session
        sess = tf.get_default_session()
        # define operation
        op = tf.trainable_variables()
        # run op
        return sess.run(op)
