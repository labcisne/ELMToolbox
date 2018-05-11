load iris_dataset.mat
X    = irisInputs';
Y    = irisTargets';
[~,Y] = max(Y,[],2); % Only one dimensional targets are supported by this method
belm  = BELM('numberOfInputNeurons', 4, 'maxNumberOfHiddenNeurons',100);
self.calculateRMSEbelm  = belm.train(X, Y);
Yhat = belm.predict(X);

load iris_dataset.mat
X    = irisInputs';
Y    = irisTargets';
[~,Y] = max(Y,[],2);    % Only one dimensional targets are supported by this method
ifelm  = CIELM('numberOfInputNeurons', 4, 'numberOfHiddenNeurons',100);
ifelm  = ifelm.train(X, Y);
Yhat = ifelm.predict(X);

load iris_dataset.mat
X    = irisInputs';
Y    = irisTargets';
eemelm  = EEMELM('numberOfInputNeurons', 4, 'maxNumberOfHiddenNeurons',100);
eemelm  = eemelm.train(X, Y);
Yhat = eemelm.predict(X);

load iris_dataset.mat
X    = irisInputs';
Y    = irisTargets';
emelm  = EMELM('numberOfInputNeurons', 4, 'numberOfHiddenNeurons',100);
emelm  = emelm.train(X, Y);
Yhat = emelm.predict(X);

load iris_dataset.mat
X    = irisInputs';
Y    = irisTargets';
ielm  = IELM('numberOfInputNeurons', 4, 'numberOfHiddenNeurons',100);
ielm  = ielm.train(X, Y);
Yhat = ielm.predict(X);

load iris_dataset.mat
X    = irisInputs';
Y    = irisTargets';
ifelm  = IFELM('numberOfInputNeurons', 4, 'numberOfHiddenNeurons',100);
ifelm  = ifelm.train(X, Y);
Yhat = ifelm.predict(X);

load iris_dataset.mat
X    = irisInputs';
Y    = irisTargets';
irelm  = IRELM('numberOfInputNeurons', 4, 'numberOfHiddenNeurons',100);
irelm  = irelm.train(X, Y);
Yhat = irelm.predict(X);


load iris_dataset.mat
X    = irisInputs';
Y    = irisTargets';
rielm  = RIELM('numberOfInputNeurons', 4, 'numberOfHiddenNeurons',100);
rielm  = rielm.train(X, Y);
Yhat = rielm.predict(X);