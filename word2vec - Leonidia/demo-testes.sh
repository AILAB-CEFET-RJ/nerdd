make
time ./word2vec -train relatos_tratado.txt -output vectors_W7_d100.txt -cbow 1 -size 100 -window 7 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 0 -iter 15
