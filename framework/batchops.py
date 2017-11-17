from nltk import word_tokenize

import numpy as np

PAD = '<pad>'
UNK = '<unk>'

'''
    maximum length of max-length sequence 
     in a batch of sequences

'''
def seq_maxlen(seqs):
    return max([len(seq) for seq in seqs])

def pad_seq(seqs, maxlen=0, PAD=PAD, truncate=False):

    # pad sequence with PAD
    #  if seqs is a list of lists
    if type(seqs[0]) == type([]):

        # get maximum length of sequence
        maxlen = maxlen if maxlen else seq_maxlen(seqs)

        def pad_seq_(seq):
            if truncate and len(seq) > maxlen:
                # truncate sequence
                return seq[:maxlen]

            # return padded
            return seq + [PAD]*(maxlen-len(seq))

        seqs = [ pad_seq_(seq) for seq in seqs ]
    
    return seqs

'''
    Data    : SQuAD single-word answers
    Network : ASReader

'''
def op1(batch, w2v):

    # fetch sequences
    context = pad_seq([ word_tokenize(p._context) 
        for p in batch ])
    # fetch the drugs
    question = pad_seq([ word_tokenize(p._question) 
        for p in batch ])
    # pad labels
    answer = np.array(pad_seq( 
        [ p._answer for p in batch],
            PAD=0), np.int32)

    return {
            'context' : w2v.encode(context),
            'question' : w2v.encode(question),
            'answer' : answer
            }

'''
    Data    : SQuAD single-word answers
    Network : DCN without decoder

'''
def op2(batch, w2v):

    # fetch sequences
    l_o_contexts = [ word_tokenize(p._context) for p in batch ]
    context = pad_seq(l_o_contexts, # maxlen = 600 -> from DCN, section 4.1
            maxlen = min(600, seq_maxlen(l_o_contexts)))
    # fetch the drugs
    question = pad_seq([ word_tokenize(p._question)
        for p in batch ])
    # pad labels
    answer = np.array(pad_seq( 
        [ p._answer for p in batch],
            PAD=0), np.int32)

    return {
            'context' : w2v.encode(context),
            'question' : w2v.encode(question),
            'answer' : answer
            }
