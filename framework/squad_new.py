"""
    Text Processing Nodes for
     Stanford Question Answering Dataset 
      (SQuAD)

"""
import json
import random

from pprint import pprint as pp
from nltk import word_tokenize
from tqdm import tqdm

TRAIN_FILE = '../dataset/train-v1.1.json'
DEV_FILE = '../dataset/dev-v1.1.json'

from collections  import namedtuple
from utilz import _namedtuple_repr_, ___asdict

RawSample  = namedtuple('RawSample', 
        [   'idx',           # unique index of dataitem
            'context',       # context  <text>
            'question',      # question <text>
            'answer',        # Answer   <NamedTuple>
            ]
        )

Answer = namedtuple('Answer',
        [   'start', # character index of start of answer
            'end',   # "         "     of end   of answer
            'text'   # answer as <text>
            ]
        )

DataItem   = namedtuple('DataItem', 
        [   'idx',           # unique index of dataitem
            'context',       # context  <text>
            'question',      # question <text>
            'answer',        # answer   <text>
            'answer_indices' # word indices of answer in context ( start <int>, end <int> )
            ]
        )

DataItem.__repre__ = _namedtuple_repr_
DataItem.___asdict = ___asdict


def read_file(filename, start=0, num_samples=100):
    """
    Read SQuAD *.json file from disk

    Args:
        filename : train or test *.json file

    Returns:
        raw text data from *.json file

    """
    with open(filename) as jfile:
        return json.load(jfile)['data']

def reduce_jsonesque_data(jdata):
    """
    Reduce JSON-esque data to (contexts, QAs)

    Args:
        jdata : raw text data in json/dict format

    Returns:
        (question-answer pairs [ <dict> ], contexts [ <text> ])

    """
    qas = []
    contexts = []
    for d in jdata:
        for p in d['paragraphs']:
            contexts.append(p['context'])
            for qa in p['qas']:
                qa['context'] = len(contexts) - 1
                qas.append(qa)

    return qas, contexts

def read_squad_file(filename):
    """
    Read file (train or dev) and return dataitems

    Args:
        filename : train or test file with PATH

    Returns:
        List of partially filled DataItems

    """
    # read JSON file
    #  extract qa pairs and contexts from JSON content
    qas, contexts = reduce_jsonesque_data( read_file(filename) )

    assert len(set([ qa['context'] for qa in qas ])) == len(contexts)

    samples = []
    for i in range(len(qas)):
        # TODO 
        #  (o) check if answer exists in context

        # build an Answer <namedtuple>
        ans_ = qas[i]['answers']
        if len(ans_) > 1: # if multiple answers exist (DEV)
            # choose the most agreed
            ans_ = ans_[choose_answer( [ a['text'] for a in ans_ ] )]
        else:
            ans_ = ans_[0]

        start = int(ans_['answer_start'])
        end   = start + len(ans_['text'])
        answer = Answer(start, end, ans_['text'])

        samples.append(
                RawSample(qas[i].get('id'), # unique index
                    contexts[int(qas[i]['context'])],            # context  <text>
                    qas[i].get('question'), # question <text>
                    answer
                    )
                )
        # check if answer exists in context
        ans, context = samples[-1].answer.text, samples[-1].context
        assert ans in context, (ans, '\n', context)

    return samples

def choose_answer(answers):
    """
    Given a list of answers from multiple evaluators
     Choose an answer most agreed

    Args:
        answers : list of answers [ <text> ]

    Returns:
        index of best answer

    """
    return answers.index( max(set(answers), key=answers.count) )
