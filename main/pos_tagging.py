import theano
import numpy as np
from theano import tensor as T
from theano import config
import time
import lasagne
from sklearn.metrics import accuracy_score
from keras.preprocessing import sequence
from lasagne_lstm_nooutput import lasagne_lstm_nooutput
from lasagne_embedding_layer_2 import lasagne_embedding_layer_2
import utils

class pos_tagging(object):

    def __init__(self, params, data):

        self.get_pos_map(data)
        self.cap = params.cap
        self.lowercase = params.lowercase
        self.featuretype = params.featuretype

        chardim = params.chardim #dimension of character network layer
        worddim = params.worddim #dimension of character embedding and word LSTM layer

        if not params.nntype == "charagram":
            self.chars = self.get_character_dict(data)
            Ce = lasagne.init.Uniform(range=0.5/len(self.chars))
            Ce_np = Ce.sample((len(self.chars),params.worddim))
            Ce = theano.shared(np.asarray(Ce_np, dtype=config.floatX))

        char = T.imatrix(); charmask = T.matrix()
        word = T.imatrix(); wordmask = T.matrix()

        idxs = T.ivector()
        Y = T.matrix()

        l_in_char = lasagne.layers.InputLayer((None, None))
        if params.nntype == "charlstm":
            l_mask_char = lasagne.layers.InputLayer(shape=(None, None))
            l_emb_char = lasagne.layers.EmbeddingLayer(l_in_char, input_size=Ce.get_value().shape[0],
                                              output_size=Ce.get_value().shape[1], W=Ce)
            l_lstm_char = lasagne.layers.LSTMLayer(l_emb_char, chardim, peepholes=True, learn_init=False,
                                              mask_input=l_mask_char)
            if not params.outgate:
                l_lstm_char = lasagne_lstm_nooutput(l_emb_char, chardim, peepholes=True, learn_init=False,
                                                   mask_input=l_mask_char)
            l_We = lasagne.layers.SliceLayer(l_lstm_char, -1, 1)
            We = lasagne.layers.get_output(l_We, {l_in_char: char, l_mask_char: charmask})
        elif params.nntype == "charagram":
            char = T.matrix()
            self.featuremap = self.get_feature_map(data, params.featuretype, params.cutoff, params.lowercase)
            print "Number of features: ", len(self.featuremap)

            l_in_char = lasagne.layers.InputLayer((None, len(self.featuremap)+1))
            if self.cap:
                l_in_char = lasagne.layers.InputLayer((None, len(self.featuremap)+2))
            l_1 = lasagne.layers.DenseLayer(l_in_char, chardim, nonlinearity=params.act)
            if params.numlayers == 1:
                l_We = lasagne.layers.DenseLayer(l_in_char, chardim, nonlinearity=params.act)
            elif params.numlayers == 2:
                l_We = lasagne.layers.DenseLayer(l_1, chardim, nonlinearity=params.act)
            else:
                raise ValueError('Only 1-2 layers are supported currently.')
            We = lasagne.layers.get_output(l_We, {l_in_char:char})
        elif params.nntype == "charcnn":
            l_emb_char = lasagne.layers.EmbeddingLayer(l_in_char, input_size=Ce.get_value().shape[0],
                                              output_size=Ce.get_value().shape[1], W=Ce)
            emb = lasagne.layers.DimshuffleLayer(l_emb_char, (0, 2, 1))
            conv_params = None
            if params.conv_type == 1:
                conv_params = [(175,2),(175,3),(175,4)]
            else:
                conv_params = [(25,1),(50,2),(75,3),(100,4),(125,5),(150,6)]
            layers = []
            for num_filters, filter_size in conv_params:
                conv = lasagne.layers.Conv1DLayer(emb, num_filters, filter_size, nonlinearity=params.act)
                pl = lasagne.layers.GlobalPoolLayer(conv,theano.tensor.max)
                pl = lasagne.layers.FlattenLayer(pl)
                layers.append(pl)
            concat = lasagne.layers.ConcatLayer(layers)
            l_We = lasagne.layers.DenseLayer(concat, num_units=chardim, nonlinearity=params.act)
            We = lasagne.layers.get_output(l_We, {l_in_char: char})
        else:
            l_We = None
            We = None

        l_in_word = lasagne.layers.InputLayer((None, None))
        l_mask_word = lasagne.layers.InputLayer(shape=(None, None))
        l_emb_word = lasagne_embedding_layer_2(l_in_word, chardim, We)

        l_lstm_wordf = lasagne.layers.LSTMLayer(l_emb_word, worddim, peepholes=True, learn_init=False,
                                              mask_input=l_mask_word)
        l_lstm_wordb = lasagne.layers.LSTMLayer(l_emb_word, worddim, peepholes=True, learn_init=False,
                                              mask_input=l_mask_word, backwards = True)

        l_reshapef = lasagne.layers.ReshapeLayer(l_lstm_wordf,(-1,worddim))
        l_reshapeb = lasagne.layers.ReshapeLayer(l_lstm_wordb,(-1,worddim))
        concat2 = lasagne.layers.ConcatLayer([l_reshapef, l_reshapeb])
        l_emb = lasagne.layers.DenseLayer(concat2, num_units=worddim, nonlinearity=lasagne.nonlinearities.tanh)
        l_out = lasagne.layers.DenseLayer(l_emb, num_units=len(self.tags), nonlinearity=lasagne.nonlinearities.softmax)
        embg = lasagne.layers.get_output(l_out, {l_in_word: word, l_mask_word: wordmask})

        embg = embg[idxs]
        prediction = T.argmax(embg, axis=1)

        self.all_params = lasagne.layers.get_all_params(l_out, trainable=True) + lasagne.layers.get_all_params(l_We, trainable=True)
        reg = 0.5*params.LC*sum(lasagne.regularization.l2(x) for x in self.all_params)

        cost = T.nnet.categorical_crossentropy(embg,Y)
        cost = T.mean(cost) + reg

        self.feedforward_function = None
        self.scoring_function = None
        self.cost_function = None
        self.train_function = None

        if params.nntype == "charlstm":
            self.feedforward_function = theano.function([char, charmask, word, wordmask, idxs], embg)
            self.scoring_function = theano.function([char, charmask, word, wordmask, idxs], prediction)
            self.cost_function = theano.function([char, charmask, word, wordmask, idxs, Y], cost)
            grads = theano.gradient.grad(cost, self.all_params)
            updates = lasagne.updates.momentum(grads, self.all_params, 0.2, momentum=0.95) #same as Ling et al.
            self.train_function = theano.function([char, charmask, word, wordmask, idxs, Y], cost, updates=updates)
        elif params.nntype == "charcnn" or params.nntype == "charagram":
            self.feedforward_function = theano.function([char, word, wordmask, idxs], embg)
            self.scoring_function = theano.function([char, word, wordmask, idxs], prediction)
            self.cost_function = theano.function([char, word, wordmask, idxs, Y], cost)
            grads = theano.gradient.grad(cost, self.all_params)
            updates = lasagne.updates.momentum(grads, self.all_params, 0.2, momentum=0.95) #same as Ling et al.
            self.train_function = theano.function([char, word, wordmask, idxs, Y], cost, updates=updates)

    def get_pos_map(self, data):
        tags = []
        for i in data:
            tags.extend(i[1])
        tags = list(set(tags))
        self.tags = {}
        for i in range(len(tags)):
            self.tags[tags[i]]=i

    def get_character_dict(self, data):
        d = {}
        ct = 0
        for i in data:
            for j in i[0]:
                j = " "+j.strip()+" "
                for k in j:
                    if k not in d:
                        d[k]=ct
                        ct += 1
        d['UUUNKKK'] = ct
        return d

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

    def prepare_data_conv(self, lis):
        return sequence.pad_sequences(lis).astype('int32')

    def get_idxs(self, xmask):
        tmp = xmask.reshape(-1,1)
        idxs = []
        for i in range(len(tmp)):
            if tmp[i] > 0:
                idxs.append(i)
        return np.asarray(idxs).astype('int32')

    def get_y(self, batch):
        ys = []
        for i in batch:
            for j in i[1]:
                y = np.zeros(len(self.tags)).astype(theano.config.floatX)
                y[self.tags[j]] = 1
                ys.append(y)
        y = np.vstack(ys) + 0.0001
        return y

    def get_word_arr(self, batch):
        all_words = []
        for i in batch:
            ph = i[0]
            all_words.extend(ph)
        all_words = list(set(all_words))
        return all_words

    def populate_embeddings_characters(self, word_matrix):
        all_embeddings = []
        for i in word_matrix:
            i = i.strip()
            i = " " + i + " "
            embeddings = []
            for j in i:
                embeddings.append(self.lookup(self.chars,j))
            all_embeddings.append(embeddings)
        return all_embeddings

    def populate_embeddings_characters_charagram(self, word_matrix):
        all_embeddings = []
        for i in word_matrix:
            all_embeddings.append(self.hash(i))
        return all_embeddings

    def populate_embeddings_words(self, batch, word_matrix):
        words = {}
        seqs = []
        for i in range(len(word_matrix)):
            words[word_matrix[i]] = i
        for i in batch:
            seqs.append(self.populate_embeddings(i[0],words))
        return seqs

    def populate_embeddings(self, ph, words):
        embeddings = []
        for i in ph:
            embeddings.append(self.lookup(words,i))
        return embeddings

    def lookup(self,words,w):
        if w in words:
            return words[w]
        else:
            return words['UUUNKKK']

    def get_feature_map(self, data, featuretype, cutoff, dolowercase):
        idx = 0
        d = {}
        feature_map = {}
        for i in data:
            for j in i[0]:
                j = " "+j.strip()+" "
                if dolowercase:
                    j = j.lower()
                if "2" in featuretype:
                    self.feature_map_helper(j,d,2)
                if "3" in featuretype:
                    self.feature_map_helper(j,d,3)
                if "4" in featuretype:
                    self.feature_map_helper(j,d,4)
                if "5" in featuretype:
                    self.feature_map_helper(j,d,5)
                if "6" in featuretype:
                    self.feature_map_helper(j,d,6)
        for i in d:
            gr = i
            ct = d[i]
            if ct >= cutoff:
                if gr in feature_map:
                    print "Error: feature overlap"
                    continue
                feature_map[gr] = idx
                idx += 1
        return feature_map

    def feature_map_helper(self,i,d,num):
        for k in range(len(i)):
            ct = k
            gram = ""
            while ct < k + num and ct < len(i):
                gram += i[ct]
                ct += 1
                if not len(gram) == num:
                    continue
                if gram in d:
                    d[gram] += 1
                else:
                    d[gram] = 1

    def get_ngram_features(self, type, word):
        features = {}
        word = word.strip()
        if self.lowercase:
            word = word.lower()
        word = " " + word + " "
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

    def has_upper(self, word):
        for i in word:
            if i.isupper():
                return True
        return False

    def update_vector(self, f, vec):
        for i in f:
            if i in self.featuremap:
                vec[self.featuremap[i]] += 1

    def hash(self, word):
        vec = np.zeros((len(self.featuremap)+1,))
        if self.cap:
            vec = np.zeros((len(self.featuremap)+2,))
        features = []
        if "2" in self.featuretype:
            features.extend(self.get_ngram_features(2,word))
        if "3" in self.featuretype:
            features.extend(self.get_ngram_features(3,word))
        if "4" in self.featuretype:
            features.extend(self.get_ngram_features(4,word))
        if "5" in self.featuretype:
            features.extend(self.get_ngram_features(5,word))
        if "6" in self.featuretype:
            features.extend(self.get_ngram_features(6,word))
        self.update_vector(features,vec)
        vec[len(self.featuremap)] = 1 #bias term
        if self.cap:
            if self.has_upper(word):
                vec[len(self.featuremap)+1] = 1 #cap term
            else:
                vec[len(self.featuremap)+1] = 0 #cap term
        vec = np.asarray(vec, dtype = config.floatX)
        return vec

    def evaluate(self, data, params):

        kf = utils.get_minibatches_idx(len(data), 100, shuffle=False)

        preds = []
        for _, train_index in kf:
            batch = [data[t] for t in train_index]
            vocab = self.get_word_arr(batch)
            x, xmask = self.prepare_data(self.populate_embeddings_words(batch, vocab))
            idxs = self.get_idxs(xmask)

            if params.nntype == "charlstm" or params.nntype == "charcnn":
                char_indices = self.populate_embeddings_characters(vocab)
            if params.nntype == "charagram":
                char_hash = self.populate_embeddings_characters_charagram(vocab)

            if params.nntype == "charlstm":
                c, cmask = self.prepare_data(char_indices)
            if params.nntype == "charcnn":
                c = self.prepare_data_conv(char_indices)

            if params.nntype == "charlstm":
                temp = self.scoring_function(c, cmask, x, xmask, idxs)
            if params.nntype == "charcnn":
                temp = self.scoring_function(c, x, xmask, idxs)
            if params.nntype == "charagram":
                temp = self.scoring_function(char_hash, x, xmask, idxs)
            preds.extend(temp)

        ys = []
        for i in data:
            for j in i[1]:
                y = self.tags[j]
                ys.append(y)

        return accuracy_score(ys, preds)

    def train(self, params, train, dev, test):
        start_time = time.time()
        counter = 0
        try:
            for eidx in xrange(params.epochs):

                kf = utils.get_minibatches_idx(len(train), params.batchsize, shuffle=True)

                uidx = 0
                for _, train_index in kf:

                    uidx += 1

                    batch = [train[t] for t in train_index]
                    vocab = self.get_word_arr(batch)
                    y = self.get_y(batch)
                    x, xmask = self.prepare_data(self.populate_embeddings_words(batch, vocab))
                    idxs = self.get_idxs(xmask)

                    if params.nntype == "charlstm" or params.nntype == "charcnn":
                        char_indices = self.populate_embeddings_characters(vocab)
                    if params.nntype == "charagram":
                        char_hash = self.populate_embeddings_characters_charagram(vocab)

                    if params.nntype == "charlstm":
                        c, cmask = self.prepare_data(char_indices)
                    if params.nntype == "charcnn":
                        c = self.prepare_data_conv(char_indices)

                    if params.nntype == "charlstm":
                        cost = self.train_function(c, cmask, x, xmask, idxs, y)
                    if params.nntype == "charcnn":
                        cost = self.train_function(c, x, xmask, idxs, y)
                    if params.nntype == "charagram":
                        cost = self.train_function(char_hash, x, xmask, idxs, y)

                    if np.isnan(cost) or np.isinf(cost):
                        print 'NaN detected'

                    #print 'Epoch ', (eidx+1), 'Update ', (uidx+1), 'Cost ', cost

                if(params.save):
                    counter += 1
                    utils.save_params(self, params.outfile+str(counter)+'.pickle')

                if(params.evaluate):
                    devscore = self.evaluate(dev, params)
                    testscore = self.evaluate(test, params)
                    trainscore = self.evaluate(train, params)
                    print "accuracy: ", devscore, testscore, trainscore

                print 'Epoch ', (eidx+1), 'Cost ', cost

        except KeyboardInterrupt:
            print "Training interrupted"

        end_time = time.time()
        print "total time:", (end_time - start_time)