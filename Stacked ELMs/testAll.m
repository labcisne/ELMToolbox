clear
close all
clc

[X,Y] = iris_dataset;
X = X';
Y = Y';

dsnelm = DSNELM('numberOfInputNeurons',4);
tic;
dsnelm = dsnelm.train(X,Y);
tdsnelm = toc;

fdsnelm = FDSNELM('numberOfInputNeurons',4);
tic;
fdsnelm = fdsnelm.train(X,Y);
tfdsnelm = toc;

selm = SELM('numberOfInputNeurons',4);
tic
selm = selm.train(X,Y);
tselm = toc;

aeselm = AESELM('numberOfInputNeurons',4);
tic;
aeselm = aeselm.train(X,Y);
taeselm = toc;