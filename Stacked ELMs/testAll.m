clear
close all
clc

[X,Y] = iris_dataset;
X = X';
Y = Y';

dsnelm = DSNELM('numberOfInputNeurons',4,'numberOfHiddenNeurons',10);
dsnelm = dsnelm.train(X,Y);

fdsnelm = FDSNELM('numberOfInputNeurons',4,'numberOfHiddenNeurons',10);
fdsnelm = fdsnelm.train(X,Y);

selm = SELM('numberOfInputNeurons',4,'numberOfHiddenNeurons',10,'reducedDimension',5);
selm = selm.train(X,Y);

aeselm = AESELM('numberOfInputNeurons',4,'numberOfHiddenNeurons',10,'reducedDimension',5);
aeselm = aeselm.train(X,Y);
