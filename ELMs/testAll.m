load iris_dataset.mat
X    = irisInputs';
Y    = irisTargets';
elm  = ELM('numberOfInputNeurons', 4, 'numberOfHiddenNeurons',100);
elm  = elm.train(X, Y);
Yhat = elm.predict(X);

load iris_dataset.mat
X    = irisInputs';
Y    = irisTargets';
grelm  = GRELM('numberOfInputNeurons', 4, 'numberOfHiddenNeurons', 100);
grelm  = grelm.train(X, Y);
Yhat = grelm.predict(X);

load iris_dataset.mat
X    = irisInputs';
Y    = irisTargets';
kelm  = KELM();
kelm  = kelm.train(X, Y);
Yhat = kelm.predict(X);

[a,b] = iris_dataset;
a = a';
b = b';
[~,b] = max(b,[],2); % MOPLPELM method only supports one dimensional outputs

moplpelm = MOPLPELM('numberOfInputNeurons',4);
moplpelm = moplpelm.train(a,b);
Yhat = moplpelm.predict(a);

% Generate a random image dataset with 10000 samples, one channel and 28 by 28 pixels;
X    = rand(28,28,1,100);
% Generate a random target vector with 3 classes;
Y    = rand(100,3);

lrfelm = LRFELM('imageSize',size(X));
lrfelm = lrfelm.train(X,Y);
Yhat = lrfelm.predict(X);

load iris_dataset.mat
X      = irisInputs';
Y      = irisTargets';
orelm  = ORELM('numberOfInputNeurons', 4);
orelm  = orelm.train(X, Y);
Yhat   = orelm.predict(X);

[a,b] = iris_dataset;
a = a';
b = b';
[~,b] = max(b,[],2); % PLPELM method only supports one dimensional outputs

plpelm = PLPELM('numberOfInputNeurons',4);
plpelm = plpelm.train(a,b);
Yhat = plpelm.predict(a);

load iris_dataset.mat
X    = irisInputs';
Y    = irisTargets';
% ridge
relm  = RELM('numberOfInputNeurons', 4, 'numberOfHiddenNeurons',100);
relm  = relm.train(X, Y);
Yhat = relm.predict(X);
% lasso
relm  = RELM('numberOfInputNeurons', 4, 'numberOfHiddenNeurons',100, 'alpha', 1);
relm  = relm.train(X, Y);
Yhat = relm.predict(X);

load iris_dataset.mat
X    = irisInputs';
Y    = irisTargets';
rkelm  = RKELM();
rkelm  = rkelm.train(X, Y);
Yhat = rkelm.predict(X);