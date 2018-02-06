clear
close all
clc

[X,Y] = iris_dataset;
X = X';
Y = Y';

orelm  = ORELM('numberOfInputNeurons', 4);
orelm = orelm.train(X,Y);

oselm  = OSELM('numberOfInputNeurons', 4, 'numberOfHiddenNeurons',100);
oselm = oselm.train(X,Y);

osrelm  = OSRELM('numberOfInputNeurons', 4, 'numberOfHiddenNeurons',100);
osrelm = osrelm.train(X,Y);

osrkelm  = OSRKELM();
osrkelm = osrkelm.train(X,Y);

reoselm  = ReOSELM('numberOfInputNeurons', 4, 'numberOfHiddenNeurons',100);
reoselm = reoselm.train(X,Y);