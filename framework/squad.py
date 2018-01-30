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

from tests import *
from info import *

TRAIN_FILE = '../dataset/train-v1.1.json'
DEV_FILE = '../dataset/dev-v1.1.json'


'''
    Read SQuAD *.json file from disk

'''
def read_file(filename, start=0, num_samples=100):
    with open(filename) as jfile:
        return json.load(jfile)['data']

'''
    Fetch the list of contexts from JSON-esque data

'''
def fetch_contexts(jdata):
    contexts = []
    for d in jdata:
        for p in d['paragraphs']:
            contexts.append(p['context'])

    return contexts

'''
    Reduce JSON-esque data to (contexts, QAs)

'''
def reduce_jsonesque_data(jdata):
    qas = []
    contexts = []
    for d in jdata:
        for p in d['paragraphs']:
            contexts.append(p['context'])
            for qa in p['qas']:
                qa['context'] = len(contexts) - 1
                qas.append(qa)

    return qas, contexts

'''
    Construct for constraint-based selection from QA-pairs

'''
def select_qa(qas, constraint_, verbose=True):
    samples = []
    for qa in qas:
        if constraint_(qa):
            # create sample
            sample = { 
                    'idx' : qa['id'], 
                    'context' : qa['context'],
                    'question' : qa['question'],
                    }
            # get [START, END] indices of answers
            indices = []
            for a in qa['answers']:
                s = a['answer_start']
                e = s + len(a['text'])
                indices.append((s,e))
            sample['answers'] = indices
            # add to list of samples
            samples.append(sample)
    if verbose:
        Gp('{}/{} = {}% samples selected'.format(
            len(samples), 
            len(qas), 
            100.*len(samples)/len(qas)))

    return samples

'''
    Check if list of answers are all one-word

'''
def one_word_constraint(x):
    answers = x['answers']
    #return len([ ai for ai in answers
    #        if len(word_tokenize(ai['text'])) > 1 ]) == 0
    return len(word_tokenize(answers[0]['text'])) == 1

'''
    Given a context, character-indexed START, END indices

     return word-indexed START, END indices

'''
def char_indices_to_word_indices(s, e, context):
    # tokenize context
    tokens = word_tokenize(context)
    # get subsequence
    sub = context[s:e]
    offset = len(word_tokenize(context[:s]))

    start = offset
    end = offset + len(word_tokenize(sub))

    assert start < len(word_tokenize(context)), (context, s, e)
    #assert end <= len(word_tokenize(context)), (context, s, e)
    if end > len(word_tokenize(context)):
        print('_______end > len__________')
        print(context, s, e)

    if len(word_tokenize(sub)) != len(tokens[start:end]):
        print('_______len not equal__________')
        print(context, s, e, start, end)

    if word_tokenize(sub)[0] != tokens[start]:
        print('_______first word not equal__________')
        print(sub)
        print(tokens[start:end])
        input()

    return start, end

'''
    Highlight answer in context

'''
def highlight_answer_by_indices(context, s, e):
    for i,w in enumerate(word_tokenize(context)):
        if i in list(range(s,e)):
            Mp(w, e=' ')
        else:
            print(w, end=' ')
    print('\n')

'''
    Highlight answer in context

'''
def highlight_answer(sample):
    if type(sample['answer']) == type(21):
        s = sample['answer'] 
        e = s+1
    elif len(sample['answer']) == 2:
        s, e = sample['answer']
    
    print(sample['question'], '\n')
    highlight_answer_by_indices(sample['context'], s, e)

'''
    Reduce text to set of words

'''
def reduce_(dataitems):
    words = []
    for item in dataitems:
        words.extend(word_tokenize(
            item['question'] + item['context']))

    return set(words)

'''
    Dump vocabulary to disk

'''
def dump_vocabulary(words):
    with open('.cache/vocabulary.txt', 'w') as f:
        for w in words:
            f.write(w)
            f.write('\n')
    # print info
    Ip(':: {} words cached'.format(len(words)))

'''
    Process
     - runs basic pipe
     - applies constraint
     - return list of dictionaries

'''
def process(filename, constraint= lambda x : True):
    # 
    # read QA-pairs, contexts
    qas, contexts = pipe0(filename)

    # select samples with one-word answers
    # qas = select_qa(qas, constraint)

    dataitems = []
    Ip(':: Processing selected samples')
    for qa in tqdm(qas):
        # (START, END) character indices of answer
        s, e = qa['answers'][0]
        # create and add sample
        dataitems.append({
            'context'  : contexts[qa['context']], # get context by id
            'question' : qa['question'],
            'answer'   : char_indices_to_word_indices(s, e, 
                contexts[qa['context']]), # character answer indices to word index
            'idx' : qa['idx']
            })

    return dataitems

'''
    Basic Pipe

'''
def pipe0(filename):
    # read from file
    #  reduce data to contexts, QA-pairs
    qas, contexts = reduce_jsonesque_data(
            read_file(filename))

    Ip(':: Read {} : {} QA pairs, {} contexts found!'.format(
        filename, len(qas), len(contexts)))

    return qas, contexts

'''
    Constraints
    
     o One-word answers

'''
def pipe1():

    def process1(filename):
        samples = []
        for sample in process(filename, one_word_constraint):
            sample['answer'] = sample['answer'][0]
            samples.append(sample)

        return samples

    # execute process - get train, test
    train = process1(TRAIN_FILE)
    test  = process1(DEV_FILE)

    # dump vocabulary
    dump_vocabulary(reduce_(train + test))

    return train, test
                
'''
    Constraints
    
     o None

'''
def pipe2():

    # execute process - get train, test
    train = process(TRAIN_FILE)
    test  = process(DEV_FILE)

    # dump vocabulary
    dump_vocabulary(reduce_(train + test))

    return train, test


if __name__ == '__main__':
    qas, contexts = pipe0(TRAIN_FILE)
    pp(qas[1246])
    pp(contexts[1246])
