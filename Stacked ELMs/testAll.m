load iris_dataset.mat
X    = irisInputs';
Y    = irisTargets';
aeselm  = AESELM('numberOfInputNeurons', 4, 'numberOfHiddenNeurons', 150);
aeselm  = aeselm.train(X, Y);
Yhat = aeselm.predict(X);

load iris_dataset.mat
X    = irisInputs';
Y    = irisTargets';
dsnelm  = DSNELM('numberOfInputNeurons', 4, 'numberOfHiddenNeurons', 100);
dsnelm  = dsnelm.train(X, Y);
Yhat = dsnelm.predict(X);

load iris_dataset.mat
X    = irisInputs';
Y    = irisTargets';
fdsnelm  = FDSNELM('numberOfInputNeurons', 4, 'numberOfHiddenNeurons', 100);
fdsnelm  = fdsnelm.train(X, Y);
Yhat = fdsnelm.predict(X);

load iris_dataset.mat
X    = irisInputs';
Y    = irisTargets';
selm  = SELM('numberOfInputNeurons', 4, 'numberOfHiddenNeurons', 150);
selm  = selm.train(X, Y);
Yhat = selm.predict(X);

