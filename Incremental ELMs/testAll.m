tic;

[a,b] = iris_dataset;
a = a';
b = b';

elm = EMELM('maxNumberOfHiddenNeurons',100,'numberOfInputNeurons',4);
elm = elm.train(a,b);

elm2 = EEMELM('numberOfInputNeurons', 4, 'maxNumberOfHiddenNeurons',100);
elm2 = elm2.train(a,b);

elm3  = CIELM('numberOfInputNeurons', 4, 'numberOfHiddenNeurons',100);
elm3  = elm3.train(a, max(b,[],2));

elm4 = BELM('numberOfInputNeurons',4,'maxNumberOfHiddenNeurons',100);
elm4 = elm4.train(a, max(b,[],2));

elm5 = IELM('numberOfInputNeurons', 4, 'numberOfHiddenNeurons',100);
elm5 = elm5.train(a,b);

elm6 = IRELM('numberOfInputNeurons', 4, 'maxNumberOfHiddenNeurons',100);
elm6 = elm6.train(a,b);

elm7 = RIELM('numberOfInputNeurons', 4, 'maxNumberOfHiddenNeurons',100);
elm7 = elm7.train(a,b);

