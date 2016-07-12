import theano
import numpy as np
from theano import tensor as T
from theano import config
import lasagne
import utils
from keras.preprocessing import sequence

class char_cnn_model(object):

    def __init__(self, params):

        self.chars = utils.get_character_dict(params.character_file)
        Ce = lasagne.init.Uniform(range=0.5/len(self.chars))
        Ce_np = Ce.sample((len(self.chars),params.chardim))
        Ce = theano.shared(np.asarray(Ce_np, dtype=config.floatX))

        g1batchindices = T.imatrix(); g2batchindices = T.imatrix()
        p1batchindices = T.imatrix(); p2batchindices = T.imatrix()

        l_in = lasagne.layers.InputLayer((None, None))
        emb = lasagne.layers.EmbeddingLayer(l_in, Ce_np.shape[0], Ce_np.shape[1], W=Ce)
        emb = lasagne.layers.DimshuffleLayer(emb, (0, 2, 1))

        conv_params = None
        if params.conv_type == 1:
            conv_params = [(175,2),(175,3),(175,4)]
        else:
            conv_params = [(25,1),(50,2),(75,3),(100,4),(125,5),(150,6)]
        layers = []
        for num_filters, filter_size in conv_params:
            conv = lasagne.layers.Conv1DLayer(emb, num_filters, filter_size, nonlinearity=params.act_conv)
            pl = lasagne.layers.GlobalPoolLayer(conv,theano.tensor.max)
            pl = lasagne.layers.FlattenLayer(pl)
            layers.append(pl)
        concat = lasagne.layers.ConcatLayer(layers)
        if params.dropout:
            concat = lasagne.layers.DropoutLayer(concat,p=params.dropout)

        l_out = lasagne.layers.DenseLayer(concat, num_units=params.worddim, nonlinearity=params.act)

        embg1 = lasagne.layers.get_output(l_out, {l_in: g1batchindices})
        embg2 = lasagne.layers.get_output(l_out, {l_in: g2batchindices})
        embp1 = lasagne.layers.get_output(l_out, {l_in: p1batchindices})
        embp2 = lasagne.layers.get_output(l_out, {l_in: p2batchindices})

        if params.dropout:
            embg1_test = lasagne.layers.get_output(l_out, {l_in: g1batchindices}, deterministic=True)
            embg2_test = lasagne.layers.get_output(l_out, {l_in: g2batchindices}, deterministic=True)

        #objective function
        g1g2 = (embg1*embg2).sum(axis=1)
        g1g2norm = T.sqrt(T.sum(embg1**2,axis=1)) * T.sqrt(T.sum(embg2**2,axis=1)) + 1E-4
        g1g2 = g1g2 / g1g2norm

        p1g1 = (embp1*embg1).sum(axis=1)
        p1g1norm = T.sqrt(T.sum(embp1**2,axis=1)) * T.sqrt(T.sum(embg1**2,axis=1)) + 1E-4
        p1g1 = p1g1 / p1g1norm

        p2g2 = (embp2*embg2).sum(axis=1)
        p2g2norm = T.sqrt(T.sum(embp2**2,axis=1)) * T.sqrt(T.sum(embg2**2,axis=1)) + 1E-4
        p2g2 = p2g2 / p2g2norm

        if params.dropout:
            g1g2_test = (embg1_test*embg2_test).sum(axis=1)
            g1g2norm_test = T.sqrt(T.sum(embg1_test**2,axis=1)) * T.sqrt(T.sum(embg2_test**2,axis=1)) + 1E-6
            g1g2_test = g1g2_test / g1g2norm_test

        costp1g1 = params.margin - g1g2 + p1g1
        costp1g1 = costp1g1*(costp1g1 > 0)

        costp2g2 = params.margin - g1g2 + p2g2
        costp2g2 = costp2g2*(costp2g2 > 0)

        cost = costp1g1 + costp2g2

        self.all_params = lasagne.layers.get_all_params(l_out, trainable=True)

        l2 = 0.
        if params.LC:
            l2 = 0.5 * params.LC * sum(lasagne.regularization.l2(x) for x in self.all_params)

        cost = T.mean(cost) + l2

        self.feedforward_function = theano.function([g1batchindices], embg1)
        self.cost_function = theano.function([g1batchindices, g2batchindices,
                                              p1batchindices, p2batchindices], cost)
        prediction = g1g2
        self.scoring_function = theano.function([g1batchindices, g2batchindices],prediction)

        if params.dropout:
            self.feedforward_function = theano.function([g1batchindices], embg1_test)
            prediction = g1g2_test
            self.scoring_function = theano.function([g1batchindices, g2batchindices],prediction)

        grads = theano.gradient.grad(cost, self.all_params)
        if params.clip:
            grads = [lasagne.updates.norm_constraint(grad, params.clip, range(grad.ndim)) for grad in grads]
        updates = params.learner(grads, self.all_params, params.eta)
        self.train_function = theano.function([g1batchindices, g2batchindices,
                                              p1batchindices, p2batchindices], cost, updates=updates)

    def prepare_data(self, lis):
        return sequence.pad_sequences(lis).astype('int32')