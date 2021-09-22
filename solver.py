import nltk, datetime, matplotlib, math, random, copy, sys, os
import dill as pickle

import numpy as np
import scipy as sc
import scipy.spatial.distance as dis

from scipy import stats
from multiprocessing import Pool

_proc = 8
_chunksize=10000
cossim_data = {}
entr_data = {}
eval_data = {}

def Cos(v1, v2):
    #print(v1, v2)
    keys = set(v1.keys()) | set(v2.keys())

    a = [ v1.get(k, 0.0) for k in keys]
    b = [ v2.get(k, 0.0) for k in keys]
    return dis.cosine(a, b)
#    norm = math.sqrt(sum(x * x for x in v1.values()) * sum(x * x for x in v2.values()))
#    ip = sum(v1[key] * v2[key] for key in v1 if key in v2)
#    return float(ip) / norm if norm else 0

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
    _x3sum = max(sum(_data2.values()), 1.0)# + _l * _vocab#len(_data2.values())

    _x = 0.0
    for i in _keys:
        _x1 = float((_data1.get(i, 0.0) ) / _x1sum)
        _x2 = float((_data2.get(i, 0.0) ) / _x2sum)

        _x -= abs(_x1 - _x2)
    return _x

def SolvWrap(_):
    return solv(_[0], _[1], _[2], _[3])

def solv(_cossim, _entr, _eval, _targ):
    #global cossim_data, entr_data, eval_data, ratio_sample
    __min_err = {}
    __alpha = {}
        
#    _cossim = cossim_data[_da]
#    _entr = entr_data[_da]
#    _eval = eval_data[_da]
#    _targ = ratio_sample[_da]
    for ratio in [0.0, 1.0, _targ]:
        _norm = max(1.0 - ratio, ratio)
        _num = len(_eval)
        _alpha = .0
        _split = 100.0
        _max = 1.0
        _min = .0
        _min_err = float('inf')
        for i in range(0, 2):
                _diff = (_max - _min) / _split
                for _a in np.arange(_min, _max, _diff):
                    _err = .0
                    for __sim, __entr, __eval in zip(_cossim, _entr, _eval):
                        _ = math.pow(( _a * __sim + (1.0 - _a ) * ( 1.0 - np.abs( __entr - ratio ) / _norm)) - __eval, 2.0 )
                        _err += _
                    if _err < _min_err:
                        _min_err = _err
                        _alpha = _a

                _max = min(1.0, _alpha + _diff)
                _min = max(0.0, _alpha - _diff)
        __min_err[ratio] = _min_err / _num
        __alpha[ratio] = _alpha

    # lambda(alpha) = 0
    _err = .0
    _a = 1.0
    for __sim, __entr, __eval in zip(_cossim, _entr, _eval):
        _ = math.pow(__sim - __eval, 2.0 )
        _err += _
    __alpha['sim'] = 1.0
    __min_err['sim'] = _err / _num

    # lambda(alpha) = 1
    _err = .0
    _a = 1.0
    for __sim, __entr, __eval in zip(_cossim, _entr, _eval):
        _ = math.pow(__entr - __eval, 2.0 )
        _err += _

    __alpha['entr'] = 0.0
    __min_err['entr'] = _err / _num

    return _da, __alpha, __min_err

def reader(i):
    import dill as pickle
    _data = {}
    with open('./result/result_'+str(i)+'.pkl', 'rb') as f:
            _ = pickle.load(f)
            for __ in _:
                try:
                    _data[__[3]].append(__)
                except:
                    _data[__[3]] = []
                    _data[__[3]].append(__)
    return i, _data
###################################################
def main():
    global cossim_data, entr_data, eval_data, ratio_sample
    try:
        if os.path.exists('./models/corpus.pkl'):
            print('Loading corpus')
            with open('./models/corpus.pkl', 'rb') as f:
                corpora = pickle.load(f)
                PROFILE = pickle.load(f)
                DA = pickle.load(f)
                ngram = pickle.load(f)
                MFC = pickle.load(f)
                MFD = pickle.load(f)
        else:
            exit(-1)
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
            exit(-1)
    except Exception as _e:
        print(_e)
        exit(-1)

    data = {}
    print('Loading result of response selection')
    for i in range(0, 1155):
        _ = reader(i)
        data[_[0]] = _[1]
#    pool = Pool(processes=_proc)
#    for _ in pool.imap_unordered(reader, range(0, 1155), chunksize=_chunksize):
#        data[_[0]] = _[1]
#    pool.close()

    cossim_data = {}
    entr_data = {}
    eval_data = {}
    for _ck, _cv in data.items():
        for _uk, _uv in _cv.items():
            for __ in _uv:
                try:
                    cossim_data[_uk].append(__[4])
                    entr_data[_uk].append(__[6])
                    eval_data[_uk].append(__[7])
                except:
                    cossim_data[_uk] = []
                    cossim_data[_uk].append(__[4])
                    entr_data[_uk] = []
                    entr_data[_uk].append(__[6])
                    eval_data[_uk] = []
                    eval_data[_uk].append(__[7])

    q = []
    for _da in DA:
        _cossim = tuple(cossim_data[_da])
        _entr = tuple(entr_data[_da])
        _eval = tuple(eval_data[_da])
        _targ = tuple(ratio_sample[_da])
        _t = (_cossim, _entr, _eval, _targ,)
        q.append(_t)

    with open('./result_alphasolve_d.xls', 'w') as f:
        f.write('DA\tAppAlpha\tAppMSE\tMinAlpha\tMinMSE\tMaxAlpha\tMaxMSE\tSimAlpha\tSimMSE\tEntrAlpha\tEntrMSE\n')

        print('Solver')
        pool = Pool(processes=_proc)
        #for _ in pool.imap_unordered(solv, DA, chunksize=_chunksize):
        for _ in pool.imap_unordered(SolvWrap, q, chunksize=_chunksize):
            print(_)
            f.write(_[0]+'\t')
            try:
                for __ in [ratio_sample[_[0]], 0.0, 1.0, 'sim', 'entr']:
                    f.write(str(_[1][__]) + '\t' + str(_[2][__]) + '\t')
            except:
                pass
            f.write('\n')
        pool.close()


if __name__ == '__main__':
    main()
