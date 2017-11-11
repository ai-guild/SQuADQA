import tensorflow as tf

from textproc import pipe1
from dataset import Dataset
from datafeed import DataFeed
from batchops import op1
from word2vocab import Word2Vocab
from trainer import Trainer

from net.tensorflow.asreader import AttentionSumReader


def exp1():
    # create dataset
    dataset = Dataset(pipe1)

    # create word2vec model
    w2v = Word2Vocab(dim=200)

    trainfeed = DataFeed(op1, batch_size=128, 
            datapoints=dataset.trainset, w2v=w2v)
    testfeed  = DataFeed(op1, batch_size=128,
            datapoints=dataset.testset, w2v=w2v)

    # instantiate model
    asreader = AttentionSumReader(hdim=200, emb_dim=200, 
            embedding=w2v.load_embeddings(), 
            vocab_size=w2v.vocab_size(), 
            num_layers=3)

    # training
    with tf.Session() as sess:
        # init variables
        sess.run(tf.global_variables_initializer())

        # create trainer instance
        trainer = Trainer(model=asreader,
                trainfeed=trainfeed,
                testfeed=testfeed,
                verbose=True,
                dump_cache=False)

        # let the training begin
        trainer.train()


if __name__ == '__main__':
    exp1()
