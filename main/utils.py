import numpy as np
from tree import tree
import time
from random import randint
from random import choice
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import cPickle
from evaluate import evaluate_wordsim
from evaluate import evaluate_sentencesim

def check_if_quarter(idx, n):
    if idx == round(n / 4.) or idx == round(n / 2.) or idx == round(3 * n / 4.):
        return True
    return False

def save_params(model, fname):
    f = file(fname, 'wb')
    cPickle.dump(model.all_params, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

def count_parameters(model):
    params = model.all_params
    ct = 0
    for i in params:
        ct += i.get_value().size
    return ct

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
        minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)

def lookup(chars, c):
    c = c.lower()
    if c in chars:
        return chars[c]
    else:
        return chars['UUUNKKK']

def get_ppdb_data(f):
    data = open(f,'r')
    lines = data.readlines()
    examples = []
    for i in lines:
        i=i.strip()
        if(len(i) > 0):
            i=i.split('\t')
            if len(i) == 2:
                e = (tree(i[0]), tree(i[1]))
                examples.append(e)
            else:
                print i
    return examples

def get_pos_data(f):
    f = open(f,'r')
    lines = f.readlines()
    sentences = []
    curr = []
    for i in lines:
        i = i.strip()
        if len(i) == 0 and len(curr) > 0:
            sentences.append(curr)
            curr = []
            continue
        i = i.split()
        wd = i[0]
        tag = i[1]
        curr.append((wd,tag))
    if len(curr) > 0:
        sentences.append(curr)
    data = []
    for i in sentences:
        words = []
        pos = []
        for j in i:
            words.append(j[0])
            pos.append(j[1])
        data.append((words,pos))
    return data

def get_pair_rand(d, idx):
    wpick = None
    ww = None
    while(wpick == None or (idx == ww)):
        ww = choice(d)
        ridx = randint(0,1)
        wpick = ww[ridx]
    return wpick

def get_pair_mix_score(d, idx, maxpair):
    r1 = randint(0,1)
    if r1 == 1:
        return maxpair
    else:
        return get_pair_rand(d,idx)

def get_pairs_fast(d, type):
    X = []
    T = []
    pairs = []
    for i in range(len(d)):
        (p1,p2) = d[i]
        X.append(p1.representation)
        X.append(p2.representation)
        T.append(p1)
        T.append(p2)

    arr = pdist(X,'cosine')
    arr = squareform(arr)

    for i in range(len(arr)):
        arr[i,i]=1
        if i % 2 == 0:
            arr[i,i+1] = 1
        else:
            arr[i,i-1] = 1

    arr = np.argmin(arr,axis=1)
    for i in range(len(d)):
        (t1,t2) = d[i]
        p1 = None
        p2 = None
        if type == "MAX":
            p1 = T[arr[2*i]]
            p2 = T[arr[2*i+1]]
        if type == "RAND":
            p1 = get_pair_rand(d,i)
            p2 = get_pair_rand(d,i)
        if type == "MIX":
            p1 = get_pair_mix_score(d,i,T[arr[2*i]])
            p2 = get_pair_mix_score(d,i,T[arr[2*i+1]])
        pairs.append((p1,p2))
    return pairs

def get_character_dict(fname):
    d = {}
    f = open(fname,'r')
    lines = f.readlines()
    ct = 0
    for i in lines:
        i = i.strip()
        if len(i) > 0:
            d[i]=ct
            ct += 1
    d[' '] = ct
    d['UUUNKKK'] = ct + 1
    return d

def get_pairs(model, batch, params):
    g1 = []; g2 = []

    if not params.nntype == "charagram":
        for i in batch:
            g1.append(i[0].embeddings)
            g2.append(i[1].embeddings)
    else:
        for i in batch:
            g1.append(model.hash(i[0].phrase))
            g2.append(model.hash(i[1].phrase))

    if params.nntype == "charlstm":
        g1x, g1mask = model.prepare_data(g1)
        g2x, g2mask = model.prepare_data(g2)
        embg1 = model.feedforward_function(g1x, g1mask)
        embg2 = model.feedforward_function(g2x, g2mask)
    elif params.nntype == "charcnn":
        g1x = model.prepare_data(g1)
        g2x = model.prepare_data(g2)
        embg1 = model.feedforward_function(g1x)
        embg2 = model.feedforward_function(g2x)
    else:
        embg1 = model.feedforward_function(g1)
        embg2 = model.feedforward_function(g2)

    for idx, i in enumerate(batch):
        i[0].representation = embg1[idx, :]
        i[1].representation = embg2[idx, :]

    pairs = get_pairs_fast(batch, params.type)
    p1 = []; p2 = []

    if not params.nntype == "charagram":
        for i in pairs:
            p1.append(i[0].embeddings)
            p2.append(i[1].embeddings)
    else:
        for i in pairs:
            p1.append(model.hash(i[0].phrase))
            p2.append(model.hash(i[1].phrase))

    if params.nntype == "charlstm":
        p1x, p1mask = model.prepare_data(p1)
        p2x, p2mask = model.prepare_data(p2)
        return (g1x, g1mask, g2x, g2mask, p1x, p1mask, p2x, p2mask)
    elif params.nntype == "charcnn":
        p1x = model.prepare_data(p1)
        p2x = model.prepare_data(p2)
        return (g1x, g2x, p1x, p2x)
    else:
        return (g1,g2,p1,p2)

def train(model, data, params):
    start_time = time.time()

    if (params.loadmodel):
        if params.domain == "word":
            evaluate_wordsim(model, params)
        elif params.domain == "sentence":
            evaluate_sentencesim(model, params)

    counter = 0
    try:
        for eidx in xrange(params.epochs):

            kf = None
            if eidx == 0 and params.shuffle1 == False:
                kf = get_minibatches_idx(len(data), params.batchsize, shuffle=False)
            else:
                kf = get_minibatches_idx(len(data), params.batchsize, shuffle=True)

            uidx = 0
            for _, train_index in kf:

                uidx += 1

                batch = [data[t] for t in train_index]

                if not params.nntype == "charagram":
                    for i in batch:
                        i[0].populate_embeddings_characters(model.chars)
                        i[1].populate_embeddings_characters(model.chars)

                if params.nntype == "charlstm":
                    (g1x, g1mask, g2x, g2mask, p1x, p1mask, p2x, p2mask) = get_pairs(model, batch, params)
                    cost = model.train_function(g1x, g2x, p1x, p2x, g1mask, g2mask, p1mask, p2mask)
                else:
                    (g1x,g2x,p1x,p2x) = get_pairs(model, batch, params)
                    cost = model.train_function(g1x, g2x, p1x, p2x)

                if np.isnan(cost) or np.isinf(cost):
                    print 'NaN detected'

                if (check_if_quarter(uidx, len(kf))):
                    if (params.save):
                        counter += 1
                        save_params(model, params.outfile + str(counter) + '.pickle')
                    if (params.evaluate):
                        if params.domain == "word":
                            evaluate_wordsim(model, params)
                        elif params.domain == "sentence":
                            evaluate_sentencesim(model, params)

                #undo batch to save RAM
                for i in batch:
                    i[0].representation = None
                    i[1].representation = None
                    if not params.nntype == "charagram":
                        i[0].unpopulate_embeddings()
                        i[1].unpopulate_embeddings()

                #print 'Epoch ', (eidx+1), 'Update ', (uidx+1), 'Cost ', cost

            if (params.save):
                counter += 1
                save_params(model, params.outfile + str(counter) + '.pickle')

            if (params.evaluate):
                if params.domain == "word":
                    evaluate_wordsim(model, params)
                elif params.domain == "sentence":
                    evaluate_sentencesim(model, params)

            print 'Epoch ', (eidx + 1), 'Cost ', cost

    except KeyboardInterrupt:
        print "Training interrupted"

    end_time = time.time()
    print "total time:", (end_time - start_time)