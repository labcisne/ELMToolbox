clear
close all
clc

[X,Y] = iris_dataset;
X = X';
Y = Y';

elm = ELM('numberOfInputNeurons',4);
elm = elm.train(X,Y);

grelm = GRELM('numberOfInputNeurons',4);
grelm = grelm.train(X,Y);

kelm = KELM();
kelm = kelm.train(X,Y);

relm = RELM('numberOfInputNeurons',4);
relm = relm.train(X,Y);

rkelm = RKELM();
rkelm = rkelm.train(X,Y);

