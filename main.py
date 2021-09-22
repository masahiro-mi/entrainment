import nltk, datetime, matplotlib, math, random, copy, sys, os
import dill as pickle

from collections import defaultdict, Counter
from swda import CorpusReader
import numpy as np
import scipy as sc
import scipy.spatial.distance as dis


import Levenshtein as LD
from scipy import stats
from multiprocessing import Pool

corpus = CorpusReader('./data/dialogue_corpora/swb/SwDA/swda')
caller_metafile = './data/dialogue_corpora/swb/SwDA/swda/call_con_tab.csv'


ACW = [
        'alright', 'gotcha', 'huh', 'mm-hm', 'okay',
        'right', 'uh-huh', 'yeah', 'yep', 'yes', 'yup'
        ]

FP = [
        'uh', 'um', 'mm'
        ]

_proc = 1
_chunksize=10000

def CalcSimWrap(_):
    return (Cos(_[1], _[2]), _[0])

def LDdist(x1, x2):
    return 1.0 - LD.distance(' '.join(x1), ' '.join(x2)) / (len(' '.join(x1)) + len(' '.join(x2)))

def Cos(v1, v2):
    #print(v1, v2)
    keys = set(v1.keys()) | set(v2.keys())

    a = [ v1.get(k, 0.0) for k in keys]
    b = [ v2.get(k, 0.0) for k in keys]
    return dis.cosine(a, b)
#    norm = math.sqrt(sum(x * x for x in v1.values()) * sum(x * x for x in v2.values()))
#    ip = sum(v1[key] * v2[key] for key in v1 if key in v2)
#    return float(ip) / norm if norm else 0

class ebdm:
    def __init__(self):
        self.e = []

    def construct_e(self):
        for _trans in corpus.iter_transcripts(display_progress=True):
            _prev = ['<init>']
            for _turn, _uttr in enumerate(_trans.utterances):
                # user profiling
                _ = {}
                _['conv_no'] = _trans.conversation_no
                _['caller'] = _uttr.caller.strip()
                _['turn'] = _turn
                _['da'] = _uttr.damsl_act_tag()
                
                _['query'] = { _k:float(len([ _ for _ in _prev if _ == _k])) for _k in list(set(_prev)) }
                #print(_['query'])
                _['response'] = [ __word.strip().lower() for __word in nltk.word_tokenize(_uttr.text) ]
                self.e.append(_)
                _prev = _['response']
        #print(self.e)

    def CalcSimWrap(self, _):
        return (_[0], _[1], self.CalcSim(_[2], _[3]),)

    def CalcSim(self, _q, _w):
        #return LDdist(_q,_w)
        return Cos(_q,_w)
    


    def respond(self, words, _no=-1, _caller=-1, _turn=-1, _da=-1, _n=1):
        #q = [ (_e, _e['query'], words,) for _e in self.e if _e['da'] == _da and ( _e['conv_no'] != _no or _e['caller'] != _caller or _e['turn'] != _turn) if len(set(words.keys()) & set(_e['query'].keys())) > 0 ]
        #q = [ (_e['response'], _e['query'], words,) for _e in self.e if _e['da'] == _da and ( _e['conv_no'] != _no or _e['caller'] != _caller or _e['turn'] != _turn) if len(set(words.keys()) & set(_e['query'].keys())) > 0 ]
        q = [ (_e['response'], _e['query'], words,) for _e in self.e if _e['da'] == _da and ( _e['conv_no'] != _no or _e['caller'] != _caller or _e['turn'] != _turn)]
        pool = Pool(processes=_proc)
        result = pool.imap(CalcSimWrap, q, chunksize=_chunksize)
        pool.close()

        #print(datetime.datetime.now(), 'respond sort')
        return sorted(list(set(result)), key=lambda x:x[0], reverse=True)[:_n]

# caler_metadata
# [conversation_no][caller] = caller_id
caller_metadata = {}
with open(caller_metafile) as f:
    for l in f:
        data = l.strip().split(',')
        try:
            if 'A' in data[1]:
                caller_metadata[int(data[0])]['A'] = int(data[2])
            if 'B' in data[1]:
                caller_metadata[int(data[0])]['B'] = int(data[2])
        except:
            caller_metadata[int(data[0])] = {}
            if 'A' in data[1]:
                caller_metadata[int(data[0])]['A'] = int(data[2])
            if 'B' in data[1]:
                caller_metadata[int(data[0])]['B'] = int(data[2])


def DataLoader():
    #data[conversation_no][speaker][part][data_type][data_element] = (counts)
    _data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
    _profile = defaultdict(lambda: defaultdict(tuple))
    #_profile[-1]['*'] = ('Average',)

    _da_list = set()
    for _trans in corpus.iter_transcripts(display_progress=True):
        for _turn, _uttr in enumerate(_trans.utterances):
            # user profiling
            _profile[_trans.conversation_no][_uttr.caller] = (caller_metadata[_trans.conversation_no][_uttr.caller],)
            _speaker = _uttr.caller.strip()
            _da = _uttr.damsl_act_tag()
            _da_list = _da_list | set([_da])
            
            for __word in nltk.word_tokenize(_uttr.text):
                _word = __word.strip().lower()
                _data[_trans.conversation_no][_speaker][_da][_word] += 1

    for i,v in _data.items():
        for _i, _v in v.items():
            for _da in _da_list:
                _data[i][_i][_da][''] = 0

    __profile = {}
    for k,v in _profile.items():
        for _k,_v in v.items():
            try:
                __profile[k][_k] = _v
            except:
                __profile[k] = {}
                __profile[k][_k] = _v

    return _data, __profile, _da_list


# effective size
def cohens_d(_data1, _data2):
    l1 = len(_data1) - 1
    l2 = len(_data2) - 1

    md = abs(np.mean(_data1) - np.mean(_data2))
    csd = l1 * np.var(_data1) + l2 * np.var(_data2)
    csd = csd / (l1 + l2)
    csd = np.sqrt(csd)

    cs = md/csd
    return cs

def pairtest(_data1, _data2):
    try:
        if len(_data1) != len(_data2):
            print('Pairwise test : Illigal data size')
            return -1

        if len(_data1) == 0 or len(_data2) == 0:
            print('Pairwise test : Null data')
            return -1

        z = 0.0
        for i in range(0, len(_data1)):
            x = _data1[i]
            y = _data2[i]

            if x > y:
                z += 1.0
    
        p = 1.0 - (z / float(len(_data1)))
        return p

    except Exception as e:
        print(e)
        return -1


def pairtest2(_data1, _data2):
    try:
        if len(_data1) != len(_data2):
            print('Pairwise test : Illigal data size')
            return -1

        if len(_data1) == 0 or len(_data2) == 0:
            print('Pairwise test : Null data')
            return -1

        ind = list(range(0, len(_data1)))
        touch_num = int(len(ind)/2)
        try_num = len(ind) * 10


        z = 0.0
        for i in range(0, try_num):
            x = 0.0
            y = 0.0

            random.shuffle(ind)

            for j in range(0, touch_num):
                x += _data1[ind[j]]
                y += _data2[ind[j]]

            if x > y:
                z += 1.0
    
        p = 1.0 - (z / float(try_num))
        return p

    except Exception as e:
        print(e)
        return -1


#######################################################
#           Entrainment score define                  #
#######################################################


def Entr1(_data1, _data2, _keys=None, _vocab=None):
    if _keys == None:
        _keys = set(_data1.keys()) | set(_data2.keys())
    _keys = set(_keys) #| set(['<unk>'])

    if _vocab == None:
        _vocab = len(set(_data1.keys()) | set(_data2.keys()))
    
    _x1sum = max(sum(_data1.values()), 1.0)# + _l * _vocab#len(_data1.values())
    _x2sum = max(sum(_data2.values()), 1.0)# + _l * _vocab#len(_data2.values())

    _x = 0.0
    for i in _keys:
        _x1 = float((_data1.get(i, 0.0) ) / _x1sum)
        _x2 = float((_data2.get(i, 0.0) ) / _x2sum)

        _x -= abs(_x1 - _x2)
    return _x

def KL_div(_data1, _data2, _keys=None, _l=1.0, _vocab=None):
    if _keys == None:
        _keys = set(_data1.keys()) | set(_data2.keys())
    _keys = set(_keys) | set(['<unk>'])

    if _vocab == None:
        _vocab = len(set(_data1.keys()) | set(_data2.keys())) + 1
    _x1sum = sum(_data1.values()) + _l * _vocab#len(_data1.values())
    _x2sum = sum(_data2.values()) + _l * _vocab#len(_data2.values())

    p = np.array( [ float((_data1.get(i, 0.0) + _l) / _x1sum) for i in _keys ] )
    q = np.array( [ float((_data2.get(i, 0.0) + _l) / _x2sum) for i in _keys ] )
    return kl(p, q)

def kl(_d1, _d2):
    return stats.entropy(_d1, _d2, 2)


def MakingNgram(corpora, _alpha=None):
    _MFC = defaultdict(int)
    _MFC_keys = defaultdict(list)
    _MFD_keys = defaultdict(list)
    _inputs = defaultdict(lambda:dict(int))

#   MFC, MFD
#   conversation_no
    for _c, _cv in corpora.items():
        _MFD = defaultdict(int)
#       speaker
        for _s, _sv in _cv.items():
#           dialogue acts
            for _d, _dv in _sv.items():
#               words
                for _k, _v in _dv.items():
                    _MFC[_k] += _v
                    _MFD[_k] += _v
                _inputs[(_c, _s, _d,)] = copy.deepcopy(_dv)

        _MFD_keys[_c]=[ __k for __i, (__k, __v) in enumerate(sorted(_MFD.items(), key=lambda x:x[1], reverse=True)) if __i < 25]

    _MFC_keys = [ __k for __i, (__k, __v) in enumerate(sorted(_MFC.items(), key=lambda x:x[1], reverse=True)) if __i < 25]
    
    return _inputs, _MFD_keys, _MFC_keys


def newt_alpha(_inputs):    
    _persons = []
    _counts = defaultdict(lambda:defaultdict(float))
    __probs = defaultdict(float)
    _probs = defaultdict(float)
    _total = defaultdict(float)

    for _prs, _arr in _inputs.items():
        if _prs[0] == -1: continue

        _persons.append(_prs)
        _sum = sum(_arr.values())

        for i, v in _arr.items():
            __probs[i] += v
            for _ in range(0, _arr[i]) : _counts[i][_] += 1.0
        for _ in range(0, _sum) : _total[_] += 1.0

    _probsum = sum(__probs.values())
    for _, __ in __probs.items() : _probs[_] = float(__/_probsum)
    
    _alpha  = .05
    _change = 1.0
    _cutoff = .000001
    # newton's method
    #print('Newton', file=sys.stderr)
    while (abs(_change) > _cutoff) :
            _lik = 0.0
            _der1 = 0.0
            _der2 = 0.0
            _val = 0.0

            for cid in _counts.keys() :
                _talpha = _probs[cid] * _alpha
                for i in _counts[cid].keys() :
                    _val = _probs[cid]/(_talpha + i)
                    _der1 += _counts[cid][i] * _val
                    _der2 -= _counts[cid][i] * _val * _val
                    _lik += _counts[cid][i] * math.log10(_talpha + i)

            for i in range(0, len(_total)) :
                _val = 1.0 / (_alpha + i)
                _der1 -= _total[i] * _val
                _der2 += _total[i] * _val * _val
                _lik -= _total[i] * math.log10(_alpha + i)

            _change = -1.0 * _der1 / _der2
            #print('newt('+str(_alpha)+') -> '+str(_lik)+' (der1='+str(_der1)+', der2='+str(_der2)+', change='+str(_change)+')', file=sys.stderr)

            if _alpha + _change > .0 : _alpha = _alpha + _change 
            else : _alpha = _alpha / 2.0
    
    for _prs, _arr in _inputs.items():
        if _prs[0] == -1: continue

        _persons.append(_prs)
        _sum = sum(_arr.values())

        for i, v in _arr.items():
            __probs[i] += v
            for _ in range(0, _arr[i]) : _counts[i][_] += 1.0
        for _ in range(0, _sum) : _total[_] += 1.0

    _probsum = sum(__probs.values())
    for _, __ in __probs.items() : _probs[_] = float(__/_probsum)
    
    # calc ngram
    _total = {}

    for i, v in _inputs.items():
        _total[i] = sum(v.values()) + _alpha
    
    return _alpha, _total, _probs

def smoothing(_inputs, _alpha, _total, _probs):
    _out = {}
    for i, v in _inputs.items():
        _total = sum(v.values()) + _alpha
        _out[i] = {}
        for j, w in v.items():
            _out[i][j] = ( _alpha * _probs[j] + w ) / _total

    return _out


def CalcCellWrap(_):
    return CalcCell(_[2], _[3], _keys=_[4]), _[0], _[1]

def CalcCell(data_x, data_y, _keys=None):
    return Entr1(data_x, data_y, _keys=_keys)
    #matrix[pos_x][pos_y] = Entr1(data_x, data_y)


def CalcCellWrap(_):
    return CalcCell(_[2], _[3], _keys=_[4]), _[0], _[1]

def CalcCell(data_x, data_y, _keys=None):
    return Entr1(data_x, data_y, _keys=_keys)
    #matrix[pos_x][pos_y] = Entr1(data_x, data_y)

def CalcIndex(ngram_x, ngram_y, _da=None, _keys=None):

    items_x = sorted([ _ for _ in ngram_x.items() if _[0][2] == _da])
    items_y = sorted([ _ for _ in ngram_y.items() if _[0][2] == _da])
    
    keys_x = sorted([ _ for _ in ngram_x.keys() if _[2] == _da])
    keys_y = sorted([ _ for _ in ngram_y.keys() if _[2] == _da])

    matrix = [ [0 for col in range(len( keys_x )) ] for row in range(len( keys_y ))]
#    q = []
    q = [ (index_id_y, index_id_x, data_x, data_y, _keys,) for index_id_x, (index_name_x, data_x) in enumerate(items_x) for index_id_y, (index_name_y, data_y) in enumerate(items_y) ]

    pool = Pool(processes=_proc)
    for _ in pool.imap(CalcCellWrap, q, chunksize=_chunksize):
        matrix[_[1]][_[2]] = _[0]
    pool.close()

    return matrix, keys_x, keys_y

def MargeCellWrap(_):
    return _[0], MargeCell(_[0], _[1], _[2], _[3])

def MargeCell(x, index_x, line_x, profile):
    if x[1] == 'A' : _x = 'B'
    else: _x = 'A'
    _tmp = []
    
    _partner = [ p for y, p in zip(index_x, line_x) if x[0] == y[0] and _x == y[1]][0]
    _tmp = [ p for y, p in zip(index_x, line_x) if profile[x[0]][x[1]] != profile[y[0]][y[1]] and profile[x[0]][x[1]] != profile[y[0]][y[1]] and profile[x[0]][_x] != profile[y[0]][y[1]] ]

    _rank = [ 1.0 for _ in _tmp if _ < _partner ] + [ .5 for _ in _tmp if _ == _partner ] +  [ .0 for _ in _tmp if _ > _partner ]

    return np.mean(_rank)

def MargeMatrix(matrix, index_x, index_y, profile):

    q = []
    #print(index_x, index_y, profile)
    for x, line_x in zip(index_y, matrix):
        if x[0] == -1: continue
        q.append(( x , index_x , line_x , profile, ))
    #print(q)
    _rank = {}
    pool = Pool(processes=_proc)
    for _ in pool.imap(MargeCellWrap, q, chunksize=_chunksize):
        _rank[_[0]] = _[1]
    pool.close()
    return _rank #{ k:np.mean(v) for k,v in _rank.items() }

    
###################################################
def main(_ts, _te):
#   EBDM base
    Ebdm = ebdm()
    try:
        if os.path.exists('./models/ebdm.pkl'):
            print('Loading ebdm')
            with open('./models/ebdm.pkl', 'rb') as f:
                Ebdm = pickle.load(f)
        else:
            print('Constructing example database')
            Ebdm.construct_e()
            with open('./models/ebdm.pkl', 'wb') as f:
                pickle.dump(Ebdm, f)
    except Exception as _e:
        print(_e)
        exit(-1)
    

    try:
        if os.path.exists('./models/corpus.pkl'):
            print('Loading corpus')
            with open('./models/corpus.pkl', 'rb') as f:
                corpora = pickle.load(f)
                PROFILE = pickle.load(f)
                DA = pickle.load(f)
                raw_ngram = pickle.load(f)
                ngram = pickle.load(f)
                alpha = pickle.load(f)
                total = pickle.load(f)
                probs = pickle.load(f)
                MFC = pickle.load(f)
                MFD = pickle.load(f)

        else:
            print('Calculating Ngram', file=sys.stderr)
            corpora, PROFILE, DA = DataLoader()
            raw_ngram, MFD, MFC = MakingNgram(corpora, _alpha=None)
            alpha, total, probs = newt_alpha(raw_ngram)
            ngram = smoothing(raw_ngram, alpha, total, probs)
            
            with open('./models/corpus.pkl', 'wb') as f:
                pickle.dump(corpora, f)
                pickle.dump(PROFILE, f)
                pickle.dump(DA, f)
                pickle.dump(raw_ngram, f)
                pickle.dump(ngram, f)
                pickle.dump(alpha, f)
                pickle.dump(total, f)
                pickle.dump(probs, f)
                pickle.dump(MFC, f)
                pickle.dump(MFD, f)

    except Exception as _e:
        print(_e)
        exit(-1)

    ratio_sample = {}
    try:
        if os.path.exists('./models/ratio_sample.pkl'):
            print('Loading true ratio')
            with open('./models/ratio_sample.pkl', 'rb') as f:
                ratio_sample = pickle.load(f)
        else:
            print('Calculating True ratio', file=sys.stderr)
            for _da in DA:
                matrix, index_x, index_y = CalcIndex(ngram, ngram, _da=_da, _keys=MFC)
                v = MargeMatrix(matrix, index_x, index_y, PROFILE)
                print(_da, np.mean(list(v.values())))
                ratio_sample[_da] = np.mean( list(v.values()) )
            with open('./models/ratio_sample.pkl', 'wb') as f:
                pickle.dump(ratio_sample, f)
    except Exception as _e:
        print(_e)
        exit(-1)

    print('Calculating training dialogue')
    xy = []
    _c = -1
    for _trans in corpus.iter_transcripts(display_progress=False):
        _data_true = defaultdict(lambda: defaultdict(int))
        _ngram = copy.deepcopy(ngram)
        prev = ['<init>']
        _c += 1
        if _c < _ts: continue
        if _c >= _te: continue

        for _turn, _uttr in enumerate(_trans.utterances):
            print(datetime.datetime.now(), _c, '/', _turn, end='\r')
            _da = _uttr.damsl_act_tag()

            #print('-=' * 20)
            #print(_uttr.text)
            _words = []

            if _uttr.caller.strip() == 'A' : _x = 'B'
            else: _x = 'A'
            #for __word in nltk.word_tokenize(_uttr.text) :
            #    _words.append(__word.strip().lower())
            #    _word = __word.strip().lower()
            _words = [ __word.strip().lower() for __word in nltk.word_tokenize(_uttr.text)]

            _prev = { _k:float(len([ _ for _ in prev if _ == _k])) for _k in list(set(prev)) }
            candidates = Ebdm.respond(_prev, _no=_trans.conversation_no, _caller=_uttr.caller.strip(), _turn=_turn, _n=20, _da=_da)

            _data = defaultdict(lambda: defaultdict(int))
            _meta_r = (_trans.conversation_no, _x, _uttr.damsl_act_tag())
            _meta_t = (_trans.conversation_no, _uttr.caller.strip(), _uttr.damsl_act_tag())
            for i, _ in enumerate(candidates):
                #print(_)
                _meta_c = (_trans.conversation_no, _uttr.caller.strip(), _uttr.damsl_act_tag(), _turn, i, )
                _data[_meta_c] = copy.deepcopy( _data_true[_meta_t] )
                for __word in _[1] : _data[_meta_c][__word] += 1
            __data = smoothing(_data, alpha, total, probs)

            _ngram_true = smoothing(_data_true, alpha, total, probs)
            _ngram[_meta_r] = _ngram_true.get(_meta_r, {})
            _ngram[_meta_t] = _ngram_true.get(_meta_t, {})


            prev = _words

            matrix, index_x, index_y = CalcIndex(_ngram, __data, _da=_da, _keys=MFC)

            v = MargeMatrix(matrix, index_x, index_y, PROFILE)
            __words = { _k:float(len([ _ for _ in _words if _ == _k])) for _k in list(set(_words)) }
            for i, _ in enumerate(candidates):
                _v = v[( _trans.conversation_no, _uttr.caller.strip(), _uttr.damsl_act_tag(), _turn, i, )]

                __ = { _k:float(len([ _a for _a in _[1] if _a == _k])) for _k in list(set(_[1])) }
                xy.append((_trans.conversation_no, _turn, _[1], _da, _[0], ratio_sample[_da], _v, Cos(__words, __),))
#                print(prev, _[1], _da, _[0], ratio_sample[_da], _v, Cos(__words, __),)


            _meta = (_trans.conversation_no, _uttr.caller.strip(), _uttr.damsl_act_tag())
            for __word in nltk.word_tokenize(_uttr.text) :
                _words.append(__word.strip().lower())
                _word = __word.strip().lower()
                _data_true[_meta][_word] += 1

        with open('./uniq_result/result_'+str(_c)+'.pkl', 'wb') as f:
            pickle.dump(xy, f)

if __name__ == '__main__':
    _ts = int(sys.argv[1])
    _te = int(sys.argv[2])
    main(_ts, _te)
