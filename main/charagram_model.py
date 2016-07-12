import theano
import numpy as np
from theano import tensor as T
from theano import config
import lasagne
import cPickle

class charagram_model(object):

    def __init__(self, params):

        self.featuremap = self.get_feature_map(params.featurefile,params.cutoff)
        print "Features:", len(self.featuremap)

        if params.loadmodel:
            p = cPickle.load(file(params.loadmodel, 'rb'))
            W = p[0]; b = p[1]

        g1feat = T.matrix(); g2feat = T.matrix()
        p1feat = T.matrix(); p2feat = T.matrix()

        l_in = lasagne.layers.InputLayer((None, len(self.featuremap)+1))
        l_1 = lasagne.layers.DenseLayer(l_in, params.worddim, nonlinearity=params.act)

        if params.numlayers == 1:
            l_end = lasagne.layers.DenseLayer(l_in, params.worddim, nonlinearity=params.act)
        elif params.numlayers == 2:
            l_end = lasagne.layers.DenseLayer(l_1, params.worddim, nonlinearity=params.act)
        else:
            raise ValueError('Only 1-2 layers are supported currently.')

        if params.loadmodel:
            l_end = lasagne.layers.DenseLayer(l_in, params.worddim, nonlinearity=params.act, W=W, b=b)

        embg1 = lasagne.layers.get_output(l_end, {l_in:g1feat})
        embg2 = lasagne.layers.get_output(l_end, {l_in:g2feat})
        embp1 = lasagne.layers.get_output(l_end, {l_in:p1feat})
        embp2 = lasagne.layers.get_output(l_end, {l_in:p2feat})

        g1g2 = (embg1*embg2).sum(axis=1)
        g1g2norm = T.sqrt(T.sum(embg1**2,axis=1)) * T.sqrt(T.sum(embg2**2,axis=1))
        g1g2 = g1g2 / g1g2norm

        p1g1 = (embp1*embg1).sum(axis=1)
        p1g1norm = T.sqrt(T.sum(embp1**2,axis=1)) * T.sqrt(T.sum(embg1**2,axis=1))
        p1g1 = p1g1 / p1g1norm

        p2g2 = (embp2*embg2).sum(axis=1)
        p2g2norm = T.sqrt(T.sum(embp2**2,axis=1)) * T.sqrt(T.sum(embg2**2,axis=1))
        p2g2 = p2g2 / p2g2norm

        costp1g1 = params.margin - g1g2 + p1g1
        costp1g1 = costp1g1*(costp1g1 > 0)

        costp2g2 = params.margin - g1g2 + p2g2
        costp2g2 = costp2g2*(costp2g2 > 0)

        cost = costp1g1 + costp2g2

        self.all_params = lasagne.layers.get_all_params(l_end, trainable=True)

        word_reg = 0.5 * params.LC * sum(lasagne.regularization.l2(x) for x in self.all_params)
        cost = T.mean(cost) + word_reg

        #feedforward
        self.feedforward_function = theano.function([g1feat], embg1)
        self.cost_function = theano.function([g1feat, g2feat,
                                              p1feat, p2feat], cost)
        prediction = g1g2
        self.scoring_function = theano.function([g1feat, g2feat],prediction)

        grads = theano.gradient.grad(cost, self.all_params)
        if params.clip:
            grads = [lasagne.updates.norm_constraint(grad, params.clip, range(grad.ndim)) for grad in grads]
        updates = params.learner(grads, self.all_params, params.eta)
        self.train_function = theano.function([g1feat, g2feat, p1feat, p2feat
                                                   ], cost, updates=updates)

    def get_feature_map(self, featurefile, cutoff):
        f = open(featurefile,'r')
        lines = f.readlines()
        idx = 0
        feature_map = {}
        for i in lines:
            i = i.split("\t")
            gr = i[0]
            gr = gr[0:len(gr)-1]
            ct = float(i[1])
            if ct >= cutoff:
                if gr in feature_map:
                    print "Error: feature overlap"
                    continue
                feature_map[gr] = idx
                idx += 1
        return feature_map

    def get_ngram_features(self, type, word):
        features = {}
        word = " "+word.strip()+" "
        for j in range(len(word)):
            idx = j
            gr = ""
            while idx < j + type and idx < len(word):
                gr += word[idx]
                idx += 1
            if not len(gr) == type:
                continue
            if gr in features:
                features[gr] += 1
            else:
                features[gr] = 1
        return features

    def update_vector(self, f, vec):
        for i in f:
            if i in self.featuremap:
                vec[self.featuremap[i]] += 1

    def hash(self, word):
        vec = np.zeros((len(self.featuremap)+1,))
        f2 = self.get_ngram_features(2,word)
        f3 = self.get_ngram_features(3,word)
        f4 = self.get_ngram_features(4,word)
        f5 = self.get_ngram_features(5,word)
        f6 = self.get_ngram_features(6,word)
        self.update_vector(f2,vec)
        self.update_vector(f3,vec)
        self.update_vector(f4,vec)
        self.update_vector(f5,vec)
        self.update_vector(f6,vec)
        vec[len(self.featuremap)] = 1 #bias term
        vec = np.asarray(vec, dtype = config.floatX)
        return vec