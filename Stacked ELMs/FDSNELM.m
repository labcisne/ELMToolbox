%   FDSNELM - Fast Deep Stacked Network constructed using SLFN modules trained with the R-ELM algorithm
%   Train and Predict a DSN based on Extreme Learning Machine, using the algorithm proposed in [1]
%
%   This code was implemented based on the following paper:
%
%   To be published...
%
%   Attributes:
%       Attributes between *.* must be informed.
%       FDSNELM objects must be created using name-value pair arguments (see the Usage Example).
%
%         *numberOfInputNeurons*:   Number of neurons in the input layer.
%                Accepted Values:   Any positive integer.
%
%          numberOfHiddenNeurons:   Number of neurons in the hidden layer
%                Accepted Values:   Any positive integer (defaut = 1000).
%
%        regularizationParameter:   Regularization Parameter (defaut = 1000)
%                Accepted Values:   Any positive real number.
%
%             maxNumberOfModules:   Number of modules of the network
%                Accepted Values:   Any positive integer number. (default = 100)
%
%             activationFunction:   Activation funcion for hidden layer
%                Accepted Values:   Function handle (see [1]) or one of these strings:
%                                       'sig':     Sigmoid (default)
%                                       'sin':     Sine
%                                       'hardlim': Hard Limit
%                                       'tribas':  Triangular basis function
%                                       'radbas':  Radial basis function
%
%                           seed:   Seed to generate the pseudo-random values.
%                                   This attribute is for reproducible research.
%                Accepted Values:   RandStream object, a integer seed for RandStream or a empty vector.
%
%       Attributes generated by the code:
%
%                    inputWeight:   Weight matrix that connects the input
%                                   layer to the hidden layer
%
%            biasOfHiddenNeurons:   Bias of hidden units
%
%                   outputWeight:   Weight matrix that connects the hidden
%                                   layer to the output layer

%                 stackedModules:   List of module objects of the network
%
%
%   Methods:
%
%        obj = FDSNELM(varargin):   Creates DSNELM objects. varargin should be in
%                                   pairs. Look attributes
%
%           obj = obj.train(X,Y):   Method for training. X is the input of size N x n,
%                                   where N is (# of samples) and n is the (# of features).
%                                   Y is the output of size N x m, where m is (# of multiple outputs)
%
%          Yhat = obj.predict(X):   Predicts the output for X.
%
%   Usage Example:
%
%       load iris_dataset.mat
%       X    = irisInputs';
%       Y    = irisTargets';
%       fdsnelm  = FDSNELM('numberOfInputNeurons', 4, 'numberOfHiddenNeurons', 100);
%       fdsnelm  = fdsnelm.train(X, Y);
%       Yhat = fdsnelm.predict(X)

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

classdef FDSNELM < Util
    properties (SetAccess = protected, GetAccess = public)
        stackedModules
        maxNumberOfModules = 100
        activationFunction = @(x) 1 ./ (1 + exp(-x));
        numberOfHiddenNeurons = 1000
        numberOfInputNeurons
        numberOfOutputNeurons
        regularizationParameter = 1000
%         seed = [];
    end
    
    methods (Access = public)
        
        function self = FDSNELM(varargin)
            
            if mod(nargin,2) ~= 0
                exception = MException('FDSNELM:ParameterError','Params must be given in pairs');
                throw (exception)
            end
            
            for i=1:2:nargin
                if isprop(self,varargin{i})
                    self.(varargin{i}) = varargin{i+1};
                else
                    exception = MException('FDSNELM:ParameterError','Given parameter does not exist');
                    throw (exception)
                end
            end
            
            self.seed = self.parseSeed();
            self.activationFunction = self.parseActivationFunction(self.activationFunction);
            
            self.stackedModules = [];
        end
        
        function [self,lastLayerOutput] = train(self,inputData,outputData)
            
            if (size(inputData,2) ~= self.numberOfInputNeurons)
                exception = MException('FDSNELM:wrongInputDimension','Wrong input dimension!');
                throw(exception);
            end
            
            auxTime = toc;
            self.numberOfOutputNeurons = size(outputData,2);
            inputHiddenWeights = -1 + 2*rand(self.seed,size(inputData,2),self.numberOfHiddenNeurons);
            biasOfHiddenNeurons = rand(self.seed,1,self.numberOfHiddenNeurons);
            lastInputDim = size(inputData,2);
            
            params = cell(1,2*9);
            params(1:2) = {'numberOfInputNeurons',self.numberOfInputNeurons};
            params(3:4) = {'numberOfHiddenNeurons',self.numberOfHiddenNeurons};
            params(5:6) = {'inputWeight',inputHiddenWeights};
            params(7:8) = {'isFirstLayer',true};
            params(9:10) = {'regularizationParameter',self.regularizationParameter};
            params(11:12) = {'activationFunction',self.activationFunction};
            params(13:14) = {'seed',self.seed};
            params(15:16) = {'numberOfOutputNeurons',self.numberOfOutputNeurons};
            params(17:18) = {'totalNumberOfInputNeurons',self.numberOfInputNeurons};
            
            newModule = FDSNELMModule(params{:});
            [newModule, lastHiddenBeforeAct,lastLayerOutput] = newModule.train(inputData,outputData,biasOfHiddenNeurons,[],[]);
            lastInputDim = lastInputDim + self.numberOfOutputNeurons;
            self.stackedModules = [self.stackedModules, newModule];
            
            while length(self.stackedModules) < self.maxNumberOfModules
                
                params(1:2) = {'numberOfInputNeurons',self.numberOfOutputNeurons};
                params(3:4) = {'numberOfHiddenNeurons',self.numberOfHiddenNeurons};
                params(5:6) = {'inputWeight',[]};
                params(7:8) = {'isFirstLayer',false};
                params(9:10) = {'regularizationParameter',self.regularizationParameter};
                params(11:12) = {'activationFunction',self.activationFunction};
                params(13:14) = {'seed',self.seed};
                params(15:16) = {'numberOfOutputNeurons',self.numberOfOutputNeurons};
                params(17:18) = {'totalNumberOfInputNeurons',lastInputDim};
                
                newModule = FDSNELMModule(params{:});
                [newModule, lastHiddenBeforeAct, lastLayerOutput] = newModule.train([],outputData,[],lastHiddenBeforeAct, lastLayerOutput);
                
                lastInputDim = lastInputDim + self.numberOfOutputNeurons;
                self.stackedModules = [self.stackedModules, newModule];
                
            end
            self.trainTime = toc - auxTime;
            
        end
        
        function pred = predict(self,inputData)
            if (size(inputData,2) ~= self.numberOfInputNeurons)
                exception = MException('FDSNELM:wrongNumberOfInputNeurons','Wrong input dimension!');
                throw(exception);
            end
            
            auxTime = toc;
            [lastLayerOutput, lastHiddenBeforeAct] = self.stackedModules(1).predict(inputData,[],[]);
            
            for i = 2:length(self.stackedModules)-1
                [lastLayerOutput, lastHiddenBeforeAct] = self.stackedModules(i).predict([],lastHiddenBeforeAct,lastLayerOutput);
                
            end
            
            pred = self.stackedModules(end).predict([],lastHiddenBeforeAct,lastLayerOutput);
            self.lastTestTime = toc - auxTime;
        end
        
        % Function used to predict the outputs in every module
        %         function predCell = predictModules(self,inputData)
        %
        %             [lastLayerOutput, lastHiddenBeforeAct] = self.stackedModules(1).predict(inputData,[],[]);
        %             predCell{1} = lastLayerOutput;
        %
        %             for i = 2:length(self.stackedModules)-1
        %                 [lastLayerOutput, lastHiddenBeforeAct] = self.stackedModules(i).predict([],lastHiddenBeforeAct,lastLayerOutput);
        %                 predCell{i} = lastLayerOutput;
        %
        %             end
        %
        %             pred = self.stackedModules(end).predict([],lastHiddenBeforeAct,lastLayerOutput);
        %             predCell{i+1} = pred;
        %
        %         end
        
        
    end
    
    
end
