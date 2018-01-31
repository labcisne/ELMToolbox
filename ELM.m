% ELM - Class Extreme Learning Machine
%   Train and Predic a SLFN based on Extreme Learning Machine
%
%   This code was implemented based on the following paper:
%
%   [1] Guang-Bin Huang, Qin-Yu Zhu, Chee-Kheong Siew, Extreme learning
%       machine: Theory and applications, Neurocomputing,Volume 70, 
%       Issues 1–3, 2006, Pages 489-501, ISSN 0925-2312, 
%       https://doi.org/10.1016/j.neucom.2005.12.126.
%       (http://www.sciencedirect.com/science/article/pii/S0925231206000385)
%
%   Attributes: 
%       Attributes between *.* must be informed.
%
%       *NumberofInputNeurons*:  Number of neurons in the input layer
%
%       NumberofHiddenNeurons:   Number of neurons in the hidden layer
%                                (defaut = 1000)
%
%       ActivationFunction:      Activation funcion for hidden layer   
%           'sig':     Sigmoid (default)
%           'sin':     Sine
%           'hardlim': Hard Limit
%           'tribas':  Triangular basis function
%           'radbas':  Radial basis function
%
%       Seed:                    RandStream object ou a seed for RandStream.
%                                This attribute is for reproducible research
%
%       InputWeight:             Weight matrix that connects the input 
%                                layer to the hidden layer
%
%       BiasofHiddenNeurons:     Bias of hidden units
%
%       OutputWeight:            Weight matrix that connects the hidden
%                                layer to the output layer
%
%   Methods:
%
%       obj = ELM(varargin):        Creates ELM objects. varargin should be in
%                                   pairs. Look attributes
%
%       obj = obj.train(X,Y):       Method for training. X is the input of size N x n,
%                                   where N is (# of samples) and n is the (# of features).
%                                   Y is the output of size N x m, where m is (# of multiple outputs)
%                            
%       Yhat = obj.predict(X):      Predicts the output for X.
%
%   Usage Example:
%
%       load iris_dataset.mat
%       X    = irisInputs';
%       Y    = irisTargets';
%       elm  = ELM('NumberofInputNeurons', 4, 'NumberofHiddenNeurons',100);
%       elm  = elm.train(X, Y);
%       Yhat = elm.predict(X)

% License:
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
%       Authors:    F. Kentaro, B. Legora Silva, D. Cosmo 
%       email:      labcisne@gmail.com
%       website:    github.com/labcisne/ELMToolbox
%       date:       Jan/2018

classdef ELM
    properties
        NumberofHiddenNeurons = 1000                
        ActivationFunction = 'sig'
        NumberofInputNeurons = []        
        InputWeight = []
        BiasofHiddenNeurons = []
        OutputWeight = []
        Seed = []
    end
    methods
        function obj = ELM(varargin)
            for i = 1:2:nargin
                obj.(varargin{i}) = varargin{i+1};
%             setProperties(obj,nargin,varargin{:})
            end
            if isnumeric(obj.Seed) && ~isempty(obj.Seed)
                obj.Seed = RandStream('mt19937ar','Seed', obj.Seed);
            elseif ~isa(obj.Seed, 'RandStream')
                obj.Seed = RandStream.getGlobalStream();
            end
            if isempty(obj.NumberofInputNeurons)
                throw(MException('EmptyNumberOfInputNeurons','Empty Number of Input Neurons'));
            end
            obj.InputWeight = rand(obj.Seed, obj.NumberofInputNeurons, obj.NumberofHiddenNeurons)*2-1;
            obj.BiasofHiddenNeurons = rand(obj.Seed, 1, obj.NumberofHiddenNeurons);
            
            if ~isa(obj.ActivationFunction,'function_handle') && ischar(obj.ActivationFunction)
                switch lower(obj.ActivationFunction)
                    case {'sig','sigmoid'}
                        %%%%%%%% Sigmoid
                        obj.ActivationFunction = @(tempH) 1 ./ (1 + exp(-tempH));
                    case {'sin','sine'}
                        %%%%%%%% Sine
                        obj.ActivationFunction = @(tempH) sin(tempH);
                    case {'hardlim'}
                        %%%%%%%% Hard Limit
                        obj.ActivationFunction = @(tempH) double(hardlim(tempH));
                    case {'tribas'}
                        %%%%%%%% Triangular basis function
                        obj.ActivationFunction = @(tempH) tribas(tempH);
                    case {'radbas'}
                        %%%%%%%% Radial basis function
                        obj.ActivationFunction = @(tempH) radbas(tempH);
                        %%%%%%%% More activation functions can be added here
                end
            else
                throw(MException('ActivationFunctionError','Error Activation Function'));
            end
        end
        function self = train(self, X, Y)
            tempH = X*self.InputWeight + repmat(self.BiasofHiddenNeurons,size(X,1),1);
            clear X;
            H = self.ActivationFunction(tempH);
            self.OutputWeight = pinv(H) * Y;
        end
        function Yhat = predict(self, X)
            tempH = X*self.InputWeight + repmat(self.BiasofHiddenNeurons,size(X,1),1);
            clear X;
            H = self.ActivationFunction(tempH);
            Yhat = H * self.OutputWeight;
        end
    end
end