import theano
import numpy as np
from theano import tensor as T
from theano import config
from lasagne_lstm_nooutput import lasagne_lstm_nooutput
import lasagne
import utils

class char_lstm_model(object):

    def __init__(self, params):

        self.chars = utils.get_character_dict(params.character_file)
        Ce = lasagne.init.Uniform(range=0.5/len(self.chars))
        Ce = Ce.sample((len(self.chars),params.chardim))
        Ce = theano.shared(np.asarray(Ce, dtype=config.floatX))

        g1batchindices = T.imatrix(); g2batchindices = T.imatrix()
        p1batchindices = T.imatrix(); p2batchindices = T.imatrix()
        g1mask = T.matrix(); g2mask = T.matrix()
        p1mask = T.matrix(); p2mask = T.matrix()

        l_in = lasagne.layers.InputLayer((None, None))
        l_mask = lasagne.layers.InputLayer(shape=(None, None))
        l_emb = lasagne.layers.EmbeddingLayer(l_in, input_size=Ce.get_value().shape[0],
                                              output_size=Ce.get_value().shape[1], W=Ce)
        l_lstm = None
        if params.outgate:
            l_lstm = lasagne.layers.LSTMLayer(l_emb, params.worddim, peepholes=params.peepholes, learn_init=False,
                                              mask_input=l_mask)
        else:
            l_lstm = lasagne_lstm_nooutput(l_emb, params.worddim, peepholes=params.peepholes, learn_init=False,
                                       mask_input=l_mask)
        l_out = lasagne.layers.SliceLayer(l_lstm, -1, 1)

        embg1 = lasagne.layers.get_output(l_out, {l_in: g1batchindices, l_mask: g1mask})
        embg2 = lasagne.layers.get_output(l_out, {l_in: g2batchindices, l_mask: g2mask})
        embp1 = lasagne.layers.get_output(l_out, {l_in: p1batchindices, l_mask: p1mask})
        embp2 = lasagne.layers.get_output(l_out, {l_in: p2batchindices, l_mask: p2mask})

        g1g2 = (embg1*embg2).sum(axis=1)
        g1g2norm = T.sqrt(T.sum(embg1**2,axis=1)) * T.sqrt(T.sum(embg2**2,axis=1)) + 1E-6
        g1g2 = g1g2 / g1g2norm

        p1g1 = (embp1*embg1).sum(axis=1)
        p1g1norm = T.sqrt(T.sum(embp1**2,axis=1)) * T.sqrt(T.sum(embg1**2,axis=1)) + 1E-6
        p1g1 = p1g1 / p1g1norm

        p2g2 = (embp2*embg2).sum(axis=1)
        p2g2norm = T.sqrt(T.sum(embp2**2,axis=1)) * T.sqrt(T.sum(embg2**2,axis=1)) + 1E-6
        p2g2 = p2g2 / p2g2norm

        costp1g1 = params.margin - g1g2 + p1g1
        costp1g1 = costp1g1*(costp1g1 > 0)

        costp2g2 = params.margin - g1g2 + p2g2
        costp2g2 = costp2g2*(costp2g2 > 0)

        cost = costp1g1 + costp2g2


        self.all_params = lasagne.layers.get_all_params(l_out, trainable=True)
        l2 = 0.5 * params.LC * sum(lasagne.regularization.l2(x) for x in self.all_params)
        cost = T.mean(cost) + l2

        self.feedforward_function = theano.function([g1batchindices, g1mask], embg1)
        self.cost_function = theano.function([g1batchindices, g2batchindices, p1batchindices, p2batchindices,
                                              g1mask, g2mask, p1mask, p2mask], cost)
        prediction = g1g2
        self.scoring_function = theano.function([g1batchindices, g2batchindices,
                                                 g1mask, g2mask], prediction)

        grads = theano.gradient.grad(cost, self.all_params)
        if params.clip:
            grads = [lasagne.updates.norm_constraint(grad, params.clip, range(grad.ndim)) for grad in grads]
        updates = params.learner(grads, self.all_params, params.eta)
        self.train_function = theano.function([g1batchindices, g2batchindices, p1batchindices, p2batchindices,
                                                   g1mask, g2mask, p1mask, p2mask], cost, updates=updates)

    def prepare_data(self, list_of_seqs):
        lengths = [len(s) for s in list_of_seqs]
        n_samples = len(list_of_seqs)
        maxlen = np.max(lengths)
        x = np.zeros((n_samples, maxlen)).astype('int32')
        x_mask = np.zeros((n_samples, maxlen)).astype(theano.config.floatX)
        for idx, s in enumerate(list_of_seqs):
            x[idx, :lengths[idx]] = s
            x_mask[idx, :lengths[idx]] = 1.
        x_mask = np.asarray(x_mask, dtype=config.floatX)
        return x, x_mask