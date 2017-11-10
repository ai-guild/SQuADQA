import sys

from info import highlighted_answer
from nltk import word_tokenize


'''
    Datapoint
        - dataitem (dict of field-value pairs)

'''
class Datapoint(object):

    def __init__(self, dataitem):
        self.dataitem = dataitem
        # unique index
        self._idx = dataitem.get('idx')
        # separate entries
        self._context = dataitem.get('context')
        self._question = dataitem.get('question')
        self._answer = dataitem.get('answer')

    def __str__(self):
        return highlighted_answer(self.dataitem)

    def context_len(self):
        return len(word_tokenize(self._context))
