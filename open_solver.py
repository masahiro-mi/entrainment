import nltk, datetime, matplotlib, math, random, copy, sys, os
import dill as pickle

import numpy as np
import scipy as sc
import scipy.spatial.distance as dis

from scipy import stats
from multiprocessing import Pool

_proc = 32
_chunksize=10000

def Cos(v1, v2):
    #print(v1, v2)
    keys = set(v1.keys()) | set(v2.keys())

    a = [ v1.get(k, 0.0) for k in keys]
    b = [ v2.get(k, 0.0) for k in keys]
    return dis.cosine(a, b)
#    norm = math.sqrt(sum(x * x for x in v1.values()) * sum(x * x for x in v2.values()))
#    ip = sum(v1[key] * v2[key] for key in v1 if key in v2)
#    return float(ip) / norm if norm else 0

# effective sizeを計算する
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

        ind = len(_data1)
        try_num = 1000 #len(ind) * 10

        z1 = 0.0
        z2 = 0.0
        for i in range(0, try_num):
            x = 0.0
            y = 0.0
            for j in range(0, ind):
                _ = random.randint(0, ind-1)
                x += _data1[_]
                y += _data2[_]

            if x > y:
                z1 += 1.0
            if x < y:
                z2 += 1.0

        p = min( 1.0 - (z1 / float(try_num)), 1.0 - (z2 / float(try_num)))

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



def Solv(_da, _targ):
    _valid = 10
    _conv_ids = []
    _turn_ids = []
    _cossim = []
    _entr = []
    _eval = []

    #for i in range(0, 15):
    for i in range(0, 1155):
        #_ = reader(i)
        with open('./uniq_result/result_'+str(i)+'.pkl', 'rb') as f:
            _ = pickle.load(f)
            for __ in _:
                if __[3] == _da:
                    #print(__)
                    _conv_ids.append(__[0])
                    _turn_ids.append(__[1])
                    _cossim.append(__[4])
                    _entr.append(__[6])
                    _eval.append(__[7])

    if len(_eval) == 0 :
        print(_da, 'error, it is no data')
    print(_da, 'data loaded:', len(_eval))

    print(_da, 'solve alpha')
    __min_err = {}
    __max_score = {}
    __max_entr = {}
    __alpha = {}
    for ratio in [0.0, 1.0, _targ]:
        _norm = max(1.0 - ratio, ratio)
        _alpha = .0
        _split = 100.0
        _max = 1.0
        _min = .0
        _min_err = float('inf')
        for i in range(0, 4):
                _diff = (_max - _min) / _split
                for _a in np.arange(_min, _max + _diff, _diff):
                    _err = .0
                    for _cid, _tid, __sim, __entr, __eval in zip(_conv_ids, _turn_ids, _cossim, _entr, _eval):
                        if _cid % _valid == 0:
                            continue
                        _ = math.pow(( _a * __sim + (1.0 - _a ) * ( 1.0 - np.abs( __entr - ratio ) / _norm)) - __eval, 2.0 )
                        _err += _
                    if _err < _min_err:
                        _min_err = _err
                        _alpha = _a

                _max = min(1.0, _alpha + _diff)
                _min = max(0.0, _alpha - _diff)


    for _key, _a in [ ('app', _alpha), ('sim', 1.0), ('entr', 0.0), ('mid', 0.5)]:
        print(_da, 'evaluation:', _key, _a)
        _err = []
        _score = []
        _entrainment = []
        _max_eval = .0
        _max_qsim = .0

        _t = 0
        for _cid, _tid, __sim, __entr, __eval in zip(_conv_ids, _turn_ids, _cossim, _entr, _eval):
            if _cid % _valid != 0: continue
            _ = math.pow(( _alpha * __sim + (1.0 - _a ) * ( 1.0 - np.abs( __entr - ratio ) / _norm)) - __eval, 2.0 )
            _q = ( _a * __sim + (1.0 - _a ) * ( 1.0 - np.abs( __entr - ratio ) / _norm))
            _err.append(_)
            if _q > _max_qsim:
                _max_eval = __eval
                _max_entr = __entr

            if _tid != _t:
                _score.append(_max_eval)
                _entrainment.append(_max_entr)
                _max_qsim = _q
                _max_eval = __eval
                _max_entr = __entr
                _t = _tid

        __alpha[_key] = _a
        __min_err[_key] = _err
        __max_score[_key] = _score
        __max_entr[_key] = _entrainment



    print(_da, 'test')
    __test = {}
    for _i in __min_err.keys():
        for _j in __min_err.keys():
            if _i != _j:
                __test[('min_err', _i, _j, )] = pairtest(__min_err[_i], __min_err[_j])
                __test[('max_score', _i, _j, )] = pairtest(__max_score[_i], __max_score[_j])
                __test[('max_entr', _i, _j, )] = pairtest(__max_entr[_i], __max_entr[_j])

    #print(_da, __alpha, __min_err, __max_score)
    return _da, __alpha, __min_err, __max_score, __max_entr, __test

def SolvWrap(_):
    print('Solver', _)
    return Solv(*_)

def reader(i):
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
#    global cossim_data, entr_data, eval_data, ratio_sample
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



    pool = Pool(processes=_proc)
    q = sorted([ (_, ratio_sample[_],) for _ in DA ], key=lambda x:x[1], reverse=True)
    r_itr = pool.map(SolvWrap, q)

    with open('./result_solver.pkl', 'wb') as f:
        pickle.dump(r_itr, f)

    with open('./result_alphasolve_open.xls', 'w') as f:
        f.write('DA\t')
        for __ in ['app', 'sim', 'entr', 'mid']:
            f.write(__ +' alpha\t')
            f.write(__ +' MSE\t')
            f.write(__ +' Score_r\t')
            f.write(__ +' Entr_r\t')
            f.write(__ +' Ratio\t')
            
        #f.write('DA\tAppAlpha\tAppMSE\tSimAlpha\tSimMSE\tEntrAlpha\tEntrMSE')
        for _k in [ 'min_err', 'max_score', 'max_entr']:
            for _i in ['app', 'sim', 'entr', 'mid']:
                for _j in ['app', 'sim', 'entr', 'mid']:
                    if _i != _j:
                        f.write(_k+'-'+_i+'-'+_j + '\t')

        f.write('\n')

        for _ in r_itr:
            f.write(_[0]+'\t')
            try:
                for __ in ['app', 'sim', 'entr', 'mid']:
                #for __ in _[1].keys():
                    f.write(str(_[1][__]) + '\t' \
                            + str(np.mean(_[2][__])) + '\t' \
                            + str(np.mean(_[3][__])) + '\t' \
                            + str(np.mean(_[4][__])) + '\t' \
                            + str(ratio_sample[_[0]]) + '\t')


                for _k in [ 'min_err', 'max_score', 'max_entr']:
                    for _i in _[1].keys():
                        for _j in _[1].keys():
                            if _i != _j:
                                f.write(str(_[5][(_k, _i, _j,)])+'\t')
            except Exception as e:
                print(e)
                pass
            f.write('\n')
        pool.close()

if __name__ == '__main__':
    main()
