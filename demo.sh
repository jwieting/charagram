#set variables to point to path of POS dev, train, and test data
export postrain=.
export posdev=.
export postest=.

#load and evaluate pre-trained models
if [ "$1" == "load-sentence-charagram" ]; then
    sh train.sh -domain sentence -nntype charagram -outfile sentence-charagram -train ../data/ppdb-xl-phrasal-preprocessed.txt -save False -evaluate True -epochs 10 -batchsize 100 -LC 1e-06 -act tanh -numlayers 1 -featurefile ../data/charagram_phrase_features_234.txt -cutoff 0 -worddim 300 -margin 0.4 -samplingtype MAX -shuffle1 True -loadmodel ../data/charagram_phrase.pickle
elif [ "$1" == "load-word-charagram" ]; then
    sh train.sh -domain word -nntype charagram -outfile simlex-charagram -train ../data/ppdb-xl-lexical-preprocessed.txt -save False -evaluate True -epochs 50 -batchsize 25 -LC 0 -act linear -numlayers 1 -featurefile ../data/charagram_features_23456.txt -cutoff 0 -worddim 300 -margin 0.4 -samplingtype MAX -shuffle1 True -loadmodel ../data/charagram.pickle

#train POS models
elif [ "$1" == "pos-charagram" ]; then
    sh train.sh -domain pos -nntype charagram -outfile pos-charagram -save False -evaluate True -epochs 50 -batchsize 100 -LC 1e-06 -act relu -numlayers 1 -featuretype 234 -cutoff 2 -chardim 150 -worddim 50 -cap True -lowercase True -traindata $postrain -devdata $posdev -testdata $postest
elif [ "$1" == "pos-charlstm" ]; then
    sh train.sh -domain pos -nntype charlstm -outfile pos-charlstm -save False -evaluate True -epochs 50 -batchsize 100 -LC 1e-05 -outgate True -chardim 150 -worddim 50 -traindata $postrain -devdata $posdev -testdata $postest
elif [ "$1" == "pos-charcnn" ]; then
    sh train.sh -domain pos -nntype charcnn -outfile pos-charcnn -save False -evaluate True -epochs 50 -batchsize 100 -LC 1e-05 -act tanh -act_conv tanh -conv_type 1 -chardim 150 -worddim 50 -traindata $postrain -devdata $posdev -testdata $postest

#train word similarity models
elif [ "$1" == "word-charagram" ]; then
    sh train.sh -domain word -nntype charagram -outfile simlex-charagram -train ../data/ppdb-xl-lexical-preprocessed.txt -save False -evaluate True -epochs 50 -batchsize 25 -LC 0 -act linear -numlayers 1 -featurefile ../data/charagram_features_23456.txt -cutoff 0 -worddim 300 -margin 0.4 -samplingtype MAX -shuffle1 True
elif [ "$1" == "word-charlstm" ]; then
    sh train.sh -domain word -nntype charlstm -outfile simlex-charlstm -train ../data/ppdb-xl-lexical-preprocessed.txt -save False -evaluate True -epochs 50 -batchsize 50 -LC 1e-06 -outgate True -character_file ../data/characters.txt -worddim 300 -chardim 300 -margin 0.4 -samplingtype MIX -shuffle1 False
elif [ "$1" == "word-charcnn" ]; then
    sh train.sh -domain word -nntype charcnn -outfile simlex-charcnn -train ../data/ppdb-xl-lexical-preprocessed.txt -save False -evaluate True -epochs 50 -batchsize 25 -LC 1e-06 -conv_type 1 -act linear -act_conv tanh -character_file ../data/characters.txt -worddim 300 -chardim 300 -margin 0.4 -samplingtype MIX -shuffle1 False

#train sentence similarity models
elif [ "$1" == "sentence-charagram" ]; then
    sh train.sh -domain sentence -nntype charagram -outfile sentence-charagram -train ../data/ppdb-xl-phrasal-preprocessed.txt -save False -evaluate True -epochs 10 -batchsize 100 -LC 1e-06 -act tanh -numlayers 1 -featurefile ../data/charagram_phrase_features_234.txt -cutoff 0 -worddim 300 -margin 0.4 -samplingtype MAX -shuffle1 True
elif [ "$1" == "sentence-charlstm" ]; then
    sh train.sh -domain sentence -nntype charlstm -outfile sentence-charlstm -train ../data/ppdb-xl-phrasal-preprocessed.txt -save False -evaluate True -epochs 10 -batchsize 100 -LC 1e-06 -outgate False -character_file ../data/characters.txt -worddim 300 -chardim 300 -margin 0.4 -samplingtype MAX -shuffle1 False
elif [ "$1" == "sentence-charcnn" ]; then
    sh train.sh -domain sentence -nntype charcnn -outfile sentence-charcnn -train ../data/ppdb-xl-phrasal-preprocessed.txt -save False -evaluate True -epochs 10 -batchsize 25 -LC 1e-06 -conv_type 2 -act tanh -act_conv relu -character_file ../data/characters.txt -worddim 300 -chardim 300 -margin 0.4 -samplingtype MAX -shuffle1 False
else
    echo "$1 not a valid option."
fi