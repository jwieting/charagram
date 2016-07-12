from params import params
import lasagne
import random
import numpy as np
import sys
import argparse
import utils
from char_cnn_model import char_cnn_model
from char_lstm_model import char_lstm_model
from charagram_model import charagram_model
from pos_tagging import pos_tagging

def str2bool(v):
    if v is None:
        return False
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise ValueError('A type that was supposed to be boolean is not boolean.')

def str2act(v):
    if v is None:
        return v
    if v.lower() == "tanh":
        return lasagne.nonlinearities.tanh
    if v.lower() == "linear":
        return lasagne.nonlinearities.linear
    if v.lower() == "relu":
        return lasagne.nonlinearities.rectify
    raise ValueError('A type that was supposed to be a learner is not.')

def learner2bool(v):
    if v is None:
        return lasagne.updates.adam
    if v.lower() == "adagrad":
        return lasagne.updates.adagrad
    if v.lower() == "adam":
        return lasagne.updates.adam
    raise ValueError('A type that was supposed to be a learner is not.')

random.seed(1)
np.random.seed(1)

params = params()

parser = argparse.ArgumentParser()
parser.add_argument("-LC", help="Lambda for composition parameters (normal training).", type=float)
parser.add_argument("-outfile", help="Output file name.")
parser.add_argument("-batchsize", help="Size of batch.", type=int)
parser.add_argument("-chardim", help="Size of character embeddings.", type=int)
parser.add_argument("-worddim", help="Size of output embedding.", type=int)
parser.add_argument("-save", help="Whether to pickle the model.")
parser.add_argument("-traindata", help="Training data file for POS tagging.")
parser.add_argument("-devdata", help="Training data file for POS tagging.")
parser.add_argument("-testdata", help="Testing data file for POS tagging.")
parser.add_argument("-peepholes", help="Whether to use peephole connections in LSTM.",default="True")
parser.add_argument("-outgate", help="Whether to use output gate in LSTM.")
parser.add_argument("-act", help="Type of activation in output layer of charagram and charcnn models.")
parser.add_argument("-act_conv", help="Type of activation in convolution in charcnn.")
parser.add_argument("-conv_type", help="Type of filter set for charcnn.", type=int)
parser.add_argument("-nntype", help="Type of neural network.")
parser.add_argument("-evaluate", help="Whether to evaluate the model during training.")
parser.add_argument("-epochs", help="Number of epochs in training.", type=int)
parser.add_argument("-clip", help="Threshold for gradient clipping.",type=int)
parser.add_argument("-eta", help="Learning rate for word/sentence tasks.", type=float, default=0.001)
parser.add_argument("-learner", help="Either AdaGrad or Adam for word/sentence tasks.", default="adam")
parser.add_argument("-numlayers", help="Number of layers in charagram model.", type=int)
parser.add_argument("-train", help="Training data file.")
parser.add_argument("-margin", help="Margin in objective function.", type=float)
parser.add_argument("-samplingtype", help="Type of sampling used.")
parser.add_argument("-num_examples", help="Number of examples to use in training. If not set, will use all examples.", type=int)
parser.add_argument("-domain", help="Either word or sentence.")
parser.add_argument("-cap", help="Whether to have a caitaliization feature in charagram model for POS tagging.")
parser.add_argument("-lowercase", help="Whether to lowercase n-grams in charagram model for POS tagging.")
parser.add_argument("-featurefile", help="File containing n-grams and their counts.")
parser.add_argument("-featuretype", help="Set of character n-grams.")
parser.add_argument("-character_file", help="List of characters for embeddings.")
parser.add_argument("-cutoff", help="Above or equal to this, features are kept.", type=int)
parser.add_argument("-dropout", help="Dropout rate.", type=float)
parser.add_argument("-shuffle1", help="Whether to shuffle data prior to first epoch.")
parser.add_argument("-loadmodel", help="Path to one of pretrained charagram models.")

args = parser.parse_args()

params.LC = args.LC
params.outfile = args.outfile
params.batchsize = args.batchsize
params.chardim = args.chardim
params.worddim = args.worddim
params.save = str2bool(args.save)
params.traindata = args.traindata
params.devdata = args.devdata
params.testdata = args.testdata
params.peepholes = str2bool(args.peepholes)
params.outgate = str2bool(args.outgate)
params.act = str2act(args.act)
params.act_conv = str2act(args.act)
params.conv_type = args.conv_type
params.nntype = args.nntype
params.evaluate = str2bool(args.evaluate)
params.epochs = args.epochs
params.learner = learner2bool(args.learner)
params.numlayers = args.numlayers
params.train = args.train
params.margin = args.margin
params.type = args.samplingtype
params.domain = args.domain
params.cap = str2bool(args.cap)
params.lowercase = str2bool(args.lowercase)
params.featurefile = args.featurefile
params.featuretype = args.featuretype
params.character_file = args.character_file
params.cutoff = args.cutoff
params.dropout = args.dropout
params.shuffle1 = str2bool(args.shuffle1)
params.loadmodel = args.loadmodel

if args.eta:
    params.eta = args.eta

if args.clip:
    if params.clip == 0:
        params.clip = None
else:
    params.clip = None

model = None
print sys.argv

if params.domain == "pos":
    trainexamples = utils.get_pos_data(params.traindata)
    devexamples = utils.get_pos_data(params.devdata)
    testexamples = utils.get_pos_data(params.testdata)

    model = pos_tagging(params, trainexamples)
    model.train(params, trainexamples, devexamples, testexamples)
else:
    examples = utils.get_ppdb_data(params.train)

    if args.num_examples:
        examples = examples[0:args.num_examples]

    if params.nntype == 'charlstm':
        model = char_lstm_model(params)
    elif params.nntype == 'charcnn':
        model = char_cnn_model(params)
    elif params.nntype == 'charagram':
        model = charagram_model(params)
    else:
        "Error no type specified"

    utils.train(model, examples, params)