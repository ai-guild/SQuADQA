from info import S, F, FI
from nltk import word_tokenize

'''
    Witness the weirdness of this function 

'''
def get_keys(jdata, space_offset=''):
    if type(jdata) == type({}):
        keys = list(jdata.keys())
        print(space_offset, keys)
        for k in keys:
            return get_keys(jdata[k], space_offset + '\t')
    if type(jdata) == type([]):
        print(space_offset, len(jdata), 'items in list')
        return get_keys(jdata[0], space_offset + '\t')

    print(jdata)

def test_loop_hist(jdata, reduce_, tdict={}):
    for d in jdata:
        for p in d['paragraphs']:
            item = reduce_(p['qas'])
            if item not in tdict:
                tdict[item] = 0
            tdict[item] += 1
    return tdict

def test_loop_hist_inner(jdata, reduce_, tdict={}):
    for d in jdata:
        for p in d['paragraphs']:
            for qa in p['qas']:
                item = reduce_(qa)
                if item not in tdict:
                    tdict[item] = 0
                tdict[item] += 1
    return tdict
 
def test_num_questions(jdata):
    # get question/answer pair count
    qcount = test_loop_hist(jdata, 
            lambda x : len(x))

    print(S('(t) Histogram\n{}'.format(qcount)))
    print(FI('(t) {} questions in total'.format(
        sum([k*v for k,v in qcount.items()]))
        ))
    print(S('(t) {} contexts'.format(
        sum(list(qcount.values())))
        ))

def test_answer_word_count(jdata):
    # get counts of answer words
    acount = test_loop_hist_inner(jdata,
            lambda x : len(word_tokenize(
                x['answers'][0]['text'])))

    count_total = sum([v for k,v in acount.items()])

    for k,v in acount.items():
        if (100*v / count_total) > 1:
            print('[{}] {} samples -> {}'.format(k , v, 100*(v)/count_total))

def test_answer_count(jdata):
    # get counts of answer words
    acount = test_loop_hist_inner(jdata,
            lambda x : len(x['answers']))

    count_total = sum([v for k,v in acount.items()])

    for k,v in acount.items():
        if (100*v / count_total) > 1:
            print('[{}] {} samples -> {}'.format(k , v, 100*(v)/count_total))

def test_question_type(jdata):
    qtype = { q:0 for q in ['what', 'why', 'how', 
        'who', 'where', 'when', 'which', 'whose'] }

    def pick_qtype(x):
        words = word_tokenize(x['question'])[:6]
        for w in words:
            if w.lower() in qtype:
                return w.lower()
        return words[0].lower()

    # get quest type count
    qtype_ = test_loop_hist_inner(jdata, pick_qtype)
    # calc total questions
    qnum = sum(list(qtype_.values()))

    qnum_known = 0
    for k in qtype.keys():
        print(k, 100*qtype_[k]/qnum, '%')
        qnum_known += qtype_[k]

    print(S('\n{}% covered by known answer types\n'.format(
        100*qnum_known/qnum)))

    for k,v in qtype_.items():
        if v > 20 and k not in qtype.keys():
            print(FI('{} {}'.format(k, v)))

#def test_single_word_answer_count(jdata):
