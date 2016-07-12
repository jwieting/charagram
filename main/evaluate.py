from scipy.stats import spearmanr
from scipy.stats import pearsonr
import numpy as np
import utils

def read_data_words(file, pop):
    file = open(file,'r')
    lines = file.readlines()
    if pop:
        lines.pop(0)
    examples = []
    for i in lines:
        i=i.strip()
        i=i.lower()
        if(len(i) > 0):
            i=i.split()
            ex = (i[0],i[1],float(i[2]))
            examples.append(ex)
    return examples

def read_data_sentences(file, pop):
    file = open(file,'r')
    lines = file.readlines()
    if pop:
        lines.pop(0)
    examples = []
    for i in lines:
        i=i.strip()
        i=i.lower()
        if(len(i) > 0):
            i=i.split('\t')
            ex = (i[0],i[1],float(i[2]))
            examples.append(ex)
    return examples

def prepare_data(p1,p2,model,params):
    if not params.nntype == "charagram":
        chars = model.chars
        X1 = []; X2 = []
        p1 = " "+p1+" "; p2 = " "+p2+" "
        for i in p1:
            X1.append(utils.lookup(chars,i))
        for i in p2:
            X2.append(utils.lookup(chars,i))
        return X1, X2
    else:
        return model.hash(p1), model.hash(p2)

def get_correlation(examples, model, params):
    preds = []
    golds = []
    feat1 = []
    feat2 = []
    ct = 0
    for i in examples:
        p1 = i[0]; p2 = i[1]; score = i[2]
        X1, X2 = prepare_data(p1,p2,model,params)
        feat1.append(X1)
        feat2.append(X2)
        ct += 1
        if ct % 100 == 0:
            if params.nntype == "charlstm":
                x1, m1 = model.prepare_data(feat1)
                x2, m2 = model.prepare_data(feat2)
                scores = model.scoring_function(x1,x2,m1,m2)
            elif params.nntype == "charcnn":
                x1 = model.prepare_data(feat1)
                x2 = model.prepare_data(feat2)
                scores = model.scoring_function(x1,x2)
            elif params.nntype == "charagram":
                x1 = np.vstack(feat1)
                x2 = np.vstack(feat2)
                scores = model.scoring_function(x1,x2)
            scores = np.squeeze(scores)
            preds.extend(scores.tolist())
            feat1 = []; feat2 = []
        golds.append(score)
    if len(feat1) > 0:
        if params.nntype == "charlstm":
            x1, m1 = model.prepare_data(feat1)
            x2, m2 = model.prepare_data(feat2)
            scores = model.scoring_function(x1,x2,m1,m2)
        elif params.nntype == "charcnn":
            x1 = model.prepare_data(feat1)
            x2 = model.prepare_data(feat2)
            scores = model.scoring_function(x1,x2)
        elif params.nntype == "charagram":
            x1 = np.vstack(feat1)
            x2 = np.vstack(feat2)
            scores = model.scoring_function(x1,x2)
        scores = np.squeeze(scores)
        preds.extend(scores.tolist())
    return pearsonr(preds,golds)[0], spearmanr(preds,golds)[0]

def evaluate_wordsim(model, params):
    ws353ex = read_data_words('../data/wordsim353.txt', True)
    ws353sim = read_data_words('../data/wordsim-sim.txt', False)
    ws353rel = read_data_words('../data/wordsim-rel.txt', False)
    simlex = read_data_words('../data/SimLex-999.txt', True)

    _, c1 = get_correlation(ws353ex, model, params)
    _, c2 = get_correlation(ws353sim, model, params)
    _, c3 = get_correlation(ws353rel, model, params)
    _, c4 = get_correlation(simlex, model, params)
    s="{0} {1} {2} {3} ws353 ws-sim ws-rel sl999".format(c1, c2, c3, c4)

    print s

def evaluate_sentencesim(model, params):
    prefix = "../data/"
    parr = []; sarr = []

    farr = ["annotated-ppdb-dev",
            "annotated-ppdb-test"]

    for i in farr:
        p,s = get_correlation(read_data_sentences(prefix+i, False), model, params)
        parr.append(p); sarr.append(s)

    s = ""
    for i,j,k in zip(parr,sarr,farr):
        s += str(i)+" "+str(j)+" "+k+" | "

    print s