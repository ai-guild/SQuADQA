from termcolor import colored as c
from nltk import word_tokenize


'''
    color-code info

'''
I = lambda x, e=None : c(x, 'blue')
M = lambda x, e=None : c(x, 'magenta')
G = lambda x, e=None : c(x, 'green')
E = lambda x, e=None : c(x, 'red')
Y = lambda x, e=None : c(x, 'yellow')

Ip = lambda x, e=None : print(c(x, 'blue'), end=e)
Mp = lambda x, e=None : print(c(x, 'magenta'), end=e)
Gp = lambda x, e=None : print(c(x, 'green'), end=e)
Ep = lambda x, e=None : print(c(x, 'red'), end=e)
Yp = lambda x, e=None : print(c(x, 'yellow'), end=e)


'''
    Tests colors

'''
S = lambda x : c(x, 'green')
F = lambda x : c(x, 'red')
FI = lambda x :c(x, 'magenta')

'''
    Highlight answer in context

     given indices (s, e)

'''
def highlight_answer_by_indices(context, s, e):
    htext = ''
    for i,w in enumerate(word_tokenize(context)):
        if i in list(range(s,e)):
            htext += M(w, e=' ') + ' '
        else:
            htext += w + ' '

    return htext + '\n'

'''
    Highlight answer in context

'''
def highlight_answer(sample):
    if type(sample['answer']) == type(21):
        s = sample['answer'] 
        e = s+1
    elif len(sample['answer']) == 2:
        s, e = sample['answer']
    
    print( sample['question'] + '\n\n' + highlight_answer_by_indices(
            sample['context'], s, e) )

'''
    Return answer-highlighted context

'''
def highlighted_answer(sample):
    if type(sample['answer']) == type(21):
        s = sample['answer'] 
        e = s+1
    elif len(sample['answer']) == 2:
        s, e = sample['answer']
    
    return sample['question'] + '\n\n' + highlight_answer_by_indices(
            sample['context'], s, e)
