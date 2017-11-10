from textproc import pipe1
from dataset import Dataset
from datafeed import DataFeed
from batchops import op1

from trainer import Trainer


def exp1():
    # create dataset
    dataset = Dataset(pipe1)

    # create word2vec model
    w2v = Word2Vocab()

    trainfeed = DataFeed(op1, batch_size=32, 
            datapoints=dataset.trainset, w2v=w2v)
    testfeed  = DataFeed(op1, batch_size=1,
            datapoints=dataset.testset, w2v=w2v)

    # instantiate model
    # asreader = AttentionSumReader(hdim=300, emb_dim=200,
    #         vocab_size=w2v.vocab_size())


    '''
    # training
    with tf.Session() as sess:
        # init variables
        sess.run(tf.global_variables_initializer())

        # create trainer instance
        trainer = Trainer(model=asreader,
                trainfeed=trainfeed,
                testfeed=testfeed,
                verbose=True,
                dump_cache=True)

        # let the training begin
        trainer.train()

    '''
