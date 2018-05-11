% ReOSELM - Regularized Online Sequential Extreme Learning Machine Class
%   Train and Predict a SLFN based on Regularized Online Sequential Extreme Learning Machine
%
%   This code was implemented based on the following paper:
%
%   [1] Hieu Trung Huynh, Yonggwan Won, Regularized online sequential 
%       learning algorithm for single-hidden layer feedforward neural 
%       networks, Pattern Recognition Letters, Volume 32, Issue 14, 2011,
%       Pages 1930-1935, ISSN 0167-8655, 
%       https://doi.org/10.1016/j.patrec.2011.07.016.
%       (http://www.sciencedirect.com/science/article/pii/S0167865511002297)
%       
%   Attributes: 
%       Attributes between *.* must be informed.
%       ReOSELM objects must be created using name-value pair arguments (see the Usage Example).
%
%         *numberOfInputNeurons*:   Number of neurons in the input layer.
%                Accepted Values:   Any positive integer.
%
%          numberOfHiddenNeurons:   Number of neurons in the hidden layer
%                Accepted Values:   Any positive integer (defaut = 1000).
%
%       regularizationParameter:   Regularization Parameter (defaut = 1000)
%                Accepted Values:   Any positive real number.
%
%           activationFunction:     Activation funcion for hidden layer   
%              Accepted Values:     Function handle (see [1]) or one of these strings:
%                                       'sig':     Sigmoid (default)
%                                       'sin':     Sine
%                                       'hardlim': Hard Limit
%                                       'tribas':  Triangular basis function
%                                       'radbas':  Radial basis function
%
%                         seed:     Seed to generate the pseudo-random values.
%                                   This attribute is for reproducible research.
%              Accepted Values:     RandStream object or a integer seed for RandStream.
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
%
%                           pMat:   Matrix used for sequentially updating  
%                                   the outputWeight matrix
%
%   Methods:
%
%        obj = ReOSELM(varargin):   Creates ReOSELM objects. varargin should be in
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
%       reoselm  = ReOSELM('numberOfInputNeurons', 4, 'numberOfHiddenNeurons',100);
%       reoselm  = reoselm.train(X, Y);
%       Yhat = reoselm.predict(X)

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
%       Authors:    F. K. Inaba, B. L. S. Silva, D. L. Cosmo 
%       email:      labcisne@gmail.com
%       website:    github.com/labcisne/ELMToolbox
%       date:       Jan/2018

classdef ReOSELM < ELM
    properties
        regularizationParameter = 1000
        pMat = []
    end
    methods
        function self = ReOSELM(varargin)
            self = self@ELM(varargin{:});
        end
        function self = train(self, X, Y)
            auxTime = toc;
            tempH = X*self.inputWeight + repmat(self.biasOfHiddenNeurons,size(X,1),1);
            H = self.activationFunction(tempH);
            clear tempH;
            if isempty(self.pMat)
                if(size(H,1)<self.numberOfHiddenNeurons)
                    self.pMat = self.regularizationParameter*eye(size(H,2)) - self.regularizationParameter * H' * pinv(eye(size(H,1))/self.regularizationParameter + H * H') * H;
                    self.outputWeight = H' * ((eye(size(H,1))/self.regularizationParameter + H * H') \ Y); % Comment this line if speed is a concern (precision might be affected)   
                else
                    self.pMat = pinv(H'*H + eye(size(H,2))/self.regularizationParameter);
                    self.outputWeight = (eye(size(H,2))/self.regularizationParameter + H' * H) \ H' * Y;   % Comment this line if speed is a concern (precision might be affected)
                end
%                 self.outputWeight = self.pMat * H'* Y; % Uncomment this line if speed is a concern (precision might be affected)
            else    
                if(size(H,1)<self.numberOfHiddenNeurons)
                    self.pMat = self.pMat - self.pMat * H' * ((eye(size(H,1)) + H * self.pMat * H') \ H) * self.pMat;                    
                else
                    self.pMat = pinv(self.pMat + H'*H);                    
                end
                self.outputWeight = self.outputWeight + self.pMat * H' * (Y - H * self.outputWeight);
            end
            self.trainTime = toc - auxTime;
        end
        
        function Yhat = predict(self, X)
            auxTime = toc;
            tempH = X*self.inputWeight + repmat(self.biasOfHiddenNeurons,size(X,1),1);
            H = self.activationFunction(tempH);
            clear tempH;
            Yhat = H * self.outputWeight;
            self.lastTestTime = toc - auxTime;
        end
    end
end