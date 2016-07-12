# charagram

Code to train models from "Charagram: Embedding Words and Sentences via Character n-grams".

The code is written in python and requires numpy, scipy, theano, keras, and the lasagne libraries.

To get started, run setup.sh to download trained models and required files such as training data, evaluation data, and feature sets. There is a demo script that takes the model that you would like to train as a command line argument (check the script to see available choices). Check main/train.py for command line options.

If you use our code for your work please cite:

@article{wieting2016charagram,
author    = {John Wieting and Mohit Bansal and Kevin Gimpel and Karen Livescu},
title     = {Charagram: Embedding Words and Sentences via Character n-grams},
journal   = {CoRR},
volume    = {abs/1607.02789},
year      = {2016}}