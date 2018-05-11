%   KFold - KFold Cross Validation Class
%   Use the K-Fold Cross Validation method to find the best parameters of a classifier using a dataset
%
%   Attributes:
%       Attributes between *.* must be informed.
%       KFold objects must be created using at least 4 parameters (numberOfFolds, classifierHandle, paramNames, paramValues)
%       See the Usage Example.
%
%               *numberOfFolds*:    Number of folds (K) to be used
%               Accepted Values:    Any integer between 2 and the number of samples of the dataset (leave one out).
%
%            *classifierHandle*:    A function handle used to construct a classifier.
%               Accepted Values:    Any classifier constructor that uses field-value pairs to construct its object.
%                                   The classifier class must inherit from the Util class
%
%                  *paramNames*:    A cell containing the names of the parameters that will be tested
%               Accepted Values:    A cell with char arrays (not strings).
%
%                 *paramValues*:    A cell containing the values of the parameters that will be tested
%               Accepted Values:    A cell with numeric arrays.
%                                   The cell should have the same length of 'paramNames', and the positions should correspond to the names
%                                   (the first position of paramValues should be the values of the first paramName)
%
%                  metricHandle:    name or instance of a Metric subclass. 
%               Accepted Values:    Any Metric subclass (Accuray, RMSE, RRMSE, aRRMSE, etc)
%
%                shuffleSamples:    boolean value indicating if the samples should be shuffled
%               Accepted Values:    true or false. (default = true)
%
%                    stratified:    boolean value indicating if the folds should have the same proportion of classes than the dataset
%                                   Only used if the "isRegression" property is false.
%               Accepted Values:    true or false. (default = true)
%
%                     seedFolds:    Number to be used as seed to shuffle the samples. If not defined, the global randStream is used.
%               Accepted Values:    An integer between 0 and 2^32 - 1.
%
%                     seedClass:    Number to be used as seed to construct the classifier. If not defined, the global randStream is used.
%               Accepted Values:    An integer between 0 and 2^32 - 1.
%
%
%   Methods:
%
%          obj = KFold(numberOfFolds,classifierHandle,paramNames,paramValues,isRegression,shuffleSamples,stratified,seedFolds,seedClass)
%                                   Creates a Kfold object. The parameters shuffleSamples, stratified, seedFolds and seedClass are optional.
%                                   See Attributes.
%
%  paramStruct = obj.start(X,Y):    Method to test all combinations of parameters given in the KFold constructor (paramNames and paramValues attributes)
%                                   in the classifier constructed by classifierHandle. Returns an struct containing three fields:
%                                   metric, paramCell and ties, corresponding to the metric obtained in the training process,
%                                   a cell containing the best parameter combination and the combinations that returned the same metric than
%                                   the best, respectively.
%                                   X is the matrix where each line is a sample and Y is a matrix where each line is a binary vector with only one
%                                   position with value 1, indicating the class of the respective sample.
%
%   Usage Example:
%
%       load iris_dataset.mat
%       X    = irisInputs';
%       Y    = irisTargets';
%
%       % first 35 samples of every class for training and using the remaining to
%       % test the classifier
%       idxTr = [1:35,51:85,101:135];
%       idxTe = setdiff(1:150,idxTr);
%
%       class = @(varargin) RELM('numberOfInputNeurons',4,varargin{:});
%       param = {'regularizationParameter'};
%       paramGrid = {2.^(-20:1:20)};
%
%       kfold = KFold(5,class,param,paramGrid,Accuracy);
%
%       pStrc = kfold.start(X(idxTr,:),Y(idxTr,:));
%
%       relm = RELM('numberOfInputNeurons',4,pStrc.paramCell{:});
%
%       relm = relm.train(X(idxTr,:),Y(idxTr,:));
%
%       yh = relm.predict(X(idxTe,:));
%       acc = Accuracy.calculate(yh,Y(idxTe,:))
%
%   License:
%
%   Permission to use, copy, or modify this software and its documentation
%   for educational and research purposes only and without fee is here
%   granted, provided that this copyright notice and the original authors'
%   names appear on all copies and supporting documentation. This program
%   shall not be used, rewritten, or adapted as the basis of a commercial
%   software or hardware product without first obtaining permission of the
%   authors. The authors make no representations about the suitability of
%   this software for any purpose. It is provided "as is" without express
%   or implied warranty.
%
%       Federal University of Espirito Santo (UFES), Brazil
%       Computers and Neural Systems Lab. (LabCISNE)
%       Authors:    B. L. S. Silva, F. K. Inaba, D. L. Cosmo
%       email:      labcisne@gmail.com
%       website:    github.com/labcisne/ELMToolbox
%       date:       Jan/2018

classdef KFold
    properties (SetAccess = private, GetAccess = public)
        paramNames
        paramValues
        classifierHandle
        numberOfFolds
        metricHandle
        shuffleSamples
        stratified
        seedFolds = []
        seedClass = []
    end
    methods (Access = private)
        function indices = getGridIndices(self)
            gridLengths = cellfun(@(x) length(x),self.paramValues);
            gridPos = cell(1,length(gridLengths));
            for i=1:length(gridLengths)
                gridPos{i} = 1:gridLengths(i);
            end
            gridIndices = cell(1,length(gridLengths));
            [gridIndices{:}] = ndgrid(gridPos{:});
            gridIndices = cellfun(@(x) x(:),gridIndices,'UniformOutput',false);
            indices = zeros(length(gridIndices{1}),length(gridLengths));
            for i=1:length(gridIndices)
                indices(:,i) = gridIndices{i};
            end
        end
    end
    methods
        function obj = KFold(numberOfFolds,classifierHandle,paramNames,paramValues,metricHandle,shuffleSamples,stratified,seedFolds,seedClass)
            
            if nargin < 4
                exception = MException('KFold:params','Not enough parameters!');
                throw (exception)
            end
            if nargin > 7
                obj.seedFolds = seedFolds;
                obj.seedClass = seedClass;
            end
            
            % Default values
            if (nargin == 4)
                shuffleSamples = true;
                stratified = true;
            elseif nargin == 5
                shuffleSamples = true;
                stratified = true;
            elseif nargin == 6
                stratified = true;
            end
            
            if isequal(class(classifierHandle),'function_handle')
                obj.classifierHandle = classifierHandle;
            else
                exception = MException('KFold:classifierHandle','classifierHandle must be an anonymous fuction');
                throw (exception)
            end
            
            if ~isa(classifierHandle(),'Util')
                exception = MException('KFold:classifierHandle','classifierHandle must be a handle to a constructor of a subclass of Util');
                throw (exception)
            end
            
            if ~isa(metricHandle,'Metric')
                exception = MException('KFold:metricHandle','metricHandle must be a subclass of Metric');
                throw (exception)
            else
                obj.metricHandle = metricHandle;
            end
            
            obj.numberOfFolds = numberOfFolds;
            obj.shuffleSamples = shuffleSamples;
            obj.stratified = stratified;
            
            if (length(paramNames) == length(paramValues))
                obj.paramNames = paramNames;
                obj.paramValues = paramValues;
            else
                exception = MException('KFold:paramsGrid','List and grid of parameters must have the same length');
                throw (exception)
            end
            
        end
        
        function paramStruct = start(self,trData,trLab)
            
            if (self.shuffleSamples)
                if isempty(self.seedFolds)
                    perm = randperm(size(trData,1));
                else
                    rStream = RandStream.create('mt19937ar','Seed', self.seedFolds);
                    perm = randperm(rStream,size(trData,1));
                end
            else
                perm = 1:size(trData,1);
            end
            
            tamFold = floor(length(perm)/self.numberOfFolds);
            
            if (~self.metricHandle.isRegressionMetric && self.stratified)
                % Generate folds with approximately the same proportion of samples in every class
                % (similar to the crossvalind function, but instead of return the fold of every sample,
                % it constructs a vector with the indices of every fold, i.e., considering folds with 30 samples,
                % perm(1:30) are the indices of samples of the first fold, perm(31:60) of the second, etc... )
                oldPerm = perm;
                permLab = trLab(perm,:);
                classes = unique(permLab,'rows');
                idxSamples = cell(1,length(classes));
                samplesPerFold = zeros(1,length(classes));
                for i=1:length(classes)
                    aux = 1 - any(bsxfun(@minus,permLab,classes(i,:)),2);
                    idxSamples{i} = find(aux == 1);
                    samplesPerFold(i) = floor(length(idxSamples{i})/self.numberOfFolds);
                end
                fold = cell(1,self.numberOfFolds);
                for k=1:self.numberOfFolds
                    fold{k} = [];
                end
                for k=1:self.numberOfFolds
                    for l=1:length(classes)
                        fold{k} = [fold{k}; idxSamples{l}((k-1)*samplesPerFold(l)+1:k*samplesPerFold(l))];
                    end
                end
                
                % Recover missing samples
                perm = cell2mat(fold);
                missingSamples = setdiff(oldPerm,perm(:));
                foldSizeDiff = tamFold - size(perm,1);
                
                missingSamples = missingSamples(1:self.numberOfFolds*(floor(length(missingSamples)/self.numberOfFolds)));
                
                if (foldSizeDiff ~= 0)
                    aux = reshape(missingSamples,foldSizeDiff,floor(length(missingSamples)/foldSizeDiff));
                    perm = [perm;aux];
                end
                
                perm = oldPerm(perm(:));
            end
            
            %             metric = zeros(1,self.numberOfFolds);
            bestMetric = self.metricHandle.worstCase;
            
            indices = self.getGridIndices;
            
            for i=1:size(indices,1)
                classParams = cell(1,2*size(indices,2));
                for j=1:size(indices,2)
                    classParams{2*j-1} = self.paramNames{j};
                    classParams{2*j} = self.paramValues{j}(indices(i,j));
                end
                
                metric = self.metricHandle.worstCase.*ones(1,self.numberOfFolds);
                
                for k=1:self.numberOfFolds
                    if isempty(self.seedClass)
                        kClassifier = self.classifierHandle(classParams{:});
                    else
                        kStream = RandStream.create('mt19937ar','Seed', self.seedClass);
                        kClassifier = self.classifierHandle('seed',kStream,classParams{:});
                    end
                    testFoldIdx = perm((k-1)*tamFold+1:k*tamFold);
                    trainFoldIdx = setdiff(perm,testFoldIdx);
                    kClassifier = kClassifier.train(trData(trainFoldIdx,:),trLab(trainFoldIdx,:));
                    pred = kClassifier.predict(trData(testFoldIdx,:));
                    if (isempty(pred)) %sometimes libsvm does not converge
                        continue;
                    end
                    metric(k) = self.metricHandle.calculate(trLab(testFoldIdx,:),pred);
                end
                
                
                if isequal(mean(metric),bestMetric)
                    paramStruct.ties = [paramStruct.ties; classParams];
                end
                
                if (self.metricHandle.isBetter(mean(metric),bestMetric))
                    paramCell = classParams;
                    bestMetric = mean(metric);
                    paramStruct.ties = [];
                end
                
            end
            
            if ~isempty(paramStruct.ties)
                warning('There were ties in the KFold process, check the ties field of the returned paramStruct');
            end
            
            paramStruct.metric = bestMetric;
            paramStruct.paramCell = [];
            if (~isempty(paramStruct.ties))
                aux = paramStruct.ties;
                paramStruct.ties = [paramStruct.paramCell; aux(1:end-1,:)];
                paramStruct.paramCell = aux(end,:);
            else
                paramStruct.paramCell = paramCell;
            end
            
        end
    end
end

