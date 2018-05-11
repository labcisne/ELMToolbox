load iris_dataset.mat
X      = irisInputs';
Y      = irisTargets';
oselm  = OSELM('numberOfInputNeurons', 4, 'numberOfHiddenNeurons',100);
oselm  = oselm.train(X, Y);
Yhat   = oselm.predict(X);

load iris_dataset.mat
X    = irisInputs';
Y    = irisTargets';
osrelm  = OSRELM('numberOfInputNeurons', 4, 'numberOfHiddenNeurons',100);
osrelm  = osrelm.train(X, Y);
Yhat = osrelm.predict(X);

load iris_dataset.mat
X        = irisInputs';
Y        = irisTargets';
osrkelm  = OSRKELM();
osrkelm  = osrkelm.train(X, Y);
Yhat     = osrkelm.predict(X);

load iris_dataset.mat
X    = irisInputs';
Y    = irisTargets';
reoselm  = ReOSELM('numberOfInputNeurons', 4, 'numberOfHiddenNeurons',100);
reoselm  = reoselm.train(X, Y);
Yhat = reoselm.predict(X);