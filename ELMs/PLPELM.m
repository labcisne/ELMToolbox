% PLPELM - Parallel Layer Perceptron Extreme Learning Machine
%   Train and Predict a Parallel Layer Perceptron Extreme Learning Machine
%
%   This code was implemented based on the following paper:
%
%   [1] L. D. Tavares, R. R. Saldanha, and D. A. G. Vieira, 
%       “Extreme learning machine with parallel layer perceptrons,” 
%       Neurocomputing, vol. 166, pp. 164–171, 2015.
%
%   Attributes:
%       Attributes between *.* must be informed.
%       PLPELM objects must be created using name-value pair arguments (see the Usage Example).
%
%         *numberOfInputNeurons*:   Number of neurons in the input layer.
%                Accepted Values:   Any positive integer.
%
%          numberOfHiddenNeurons:   Number of neurons in each hidden layer
%                Accepted Values:   Any positive integer (defaut = 1000).
%
%                           seed:   Seed to generate the pseudo-random values.
%                                   This attribute is for reproducible research.
%                Accepted Values:   RandStream object or a integer seed for RandStream.
%
%       Attributes generated by the code:
%
%                        Vmatrix:   Weight matrix that connects the input
%                                   layer to the hidden layer
%
%                        PMatrix:   "Output weights". See [1]
%
%   Methods:
%
%       obj = PLPELM(varargin):        Creates RELM objects. varargin should be in
%                                    pairs. Look attributes
%
%         obj = obj.train(X,Y):        Method for training. X is the input of size N x n,
%                                    where N is (# of samples) and n is the (# of features).
%                                    Y is the output of size N x m, where m is (# of multiple outputs)
%
%        Yhat = obj.predict(X):       Predicts the output for X.
%
%   Usage Example:
%
%       [a,b] = iris_dataset;
%       a = a';
%       b = b';
%       [~,b] = max(b,[],2); % PLPELM method only supports one dimensional outputs
%       
%       plpelm = PLPELM('numberOfInputNeurons',4);
%       plpelm = plpelm.train(a,b);
%       Yhat = plpelm.predict(a);
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
%       date:       Feb/2018


classdef PLPELM < Util
    properties
        numberOfHiddenNeurons = 1000
        activationFunction = 'sig'
        numberOfInputNeurons = []
        Vmatrix
        Pmatrix
    end
    methods
        function self = PLPELM(varargin)
            for i = 1:2:nargin
                self.(varargin{i}) = varargin{i+1};
            end
            if isempty(self.numberOfInputNeurons)
                throw(MException('PLPELM:emptynumberOfInputNeurons','Empty Number of Input Neurons'));
            end
            
            self.seed = self.parseSeed();
            self.activationFunction = self.parseActivationFunction(self.activationFunction);
            
            self.Vmatrix = rand(self.seed, self.numberOfInputNeurons+1, self.numberOfHiddenNeurons)*2-1;
            
        end
        
        function self = train(self,X,Y)
            auxTime = toc;
            if (size(Y,2) ~= 1)
                throw(MException('PLPELM:outputSize','PLPELM method only supports one dimensional outputs'));
            end
            
            X2 = [X, ones(size(X,1),1)];
            %             V = -1 + 2*rand(size(X2,2),h);
            B = self.activationFunction(X2*self.Vmatrix);
            C = kron(B,ones(1,size(X2,2))).*repmat(X2,[1 self.numberOfHiddenNeurons]);
            
            if (size(C,1) >= size(C,2))
                self.Pmatrix = reshape(pinv(C' * C) * C' * Y,size(self.Vmatrix));
            else
                self.Pmatrix = reshape(C' * (pinv(C * C') * Y),size(self.Vmatrix));
            end

            self.trainTime = toc - auxTime;
        end
        
        function yh = predict(self,X)
            X2 = [X, ones(size(X,1),1)];
            a = X2*self.Pmatrix;
            b = self.activationFunction(X2*self.Vmatrix);
            yh = sum(a.*b,2);
        end
    end
end