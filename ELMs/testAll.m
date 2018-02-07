clear
close all
clc

[X,Y] = iris_dataset;
X = X';
Y = Y';

elm = ELM('numberOfInputNeurons',4);
elm = elm.train(X,Y);

orelm  = ORELM('numberOfInputNeurons', 4,'regularizationParameter',4);
orelm = orelm.train(X,Y);

grelm = GRELM('numberOfInputNeurons',4,'alpha',0.6);
grelm = grelm.train(X,Y);

kelm = KELM();
kelm = kelm.train(X,Y);

relm = RELM('numberOfInputNeurons',4);
relm = relm.train(X,Y);

rkelm = RKELM();
rkelm = rkelm.train(X,Y);

