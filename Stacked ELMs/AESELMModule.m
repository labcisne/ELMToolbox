%   AESELMModule - Autoencoder Stacked Extreme Learning Machine Module Class
%   Train and Predict an Autoencoder Module of a Stacked Extreme Learning Machine
%
%   This code was implemented based on the following paper:
%
%   [1] Zhou, H., Huang, G.-B., Lin, Z., Wang, H., & Soh, Y. C. (2014). 
%       Stacked Extreme Learning Machines. 
%       IEEE Transactions on Cybernetics, PP(99), 1. 
%       https://doi.org/10.1109/TCYB.2014.2363492
%       (http://ieeexplore.ieee.org/document/6937189/)
%
%   This class should not be used outside the AESELM class.
%   This class is similar to the S-ELM, but uses the autoencoder algorithm described in [1].
%   See the AESELM class help.

classdef AESELMModule < SELMModule
    
    methods (Access = ?AESELM)
        
        function obj = AESELMModule(varargin)
            obj = obj@SELMModule(varargin{:});
        end
        
    end
    
    methods
        
        function [self, projectedOutput] = train(self,inputData,outputData,lastHiddenOutput)
            if (size(inputData,2) ~= self.numberOfInputNeurons)
                exception = MException('AESELMModule:wrongInputDimension','Wrong input dimension!');
                throw(exception);
            end
            
            Hnew = self.activationFunction(inputData*self.inputWeight + repmat(self.biasOfHiddenNeurons,[size(inputData,1),1]));
            self.inputWeight = [];
            
            % The outputWeight of the auxiliar slfn is used as inputweight of the slfn that will be used:
            if size(Hnew,1)>=size(Hnew,2)
                self.inputWeight = ((eye(size(Hnew,2))/self.regularizationParameter + Hnew' * Hnew) \ Hnew' * inputData)';
            else
                self.inputWeight = (Hnew' * ((eye(size(Hnew,1))/self.regularizationParameter + Hnew * Hnew') \ inputData))';
            end
            
            HAE = self.activationFunction(inputData*self.inputWeight + repmat(self.biasOfHiddenNeurons,[size(inputData,1),1]));
            
            if ~(self.isFirstLayer)
                H = [lastHiddenOutput, HAE];
            else
                H = HAE;
            end
            
            if size(H,1)>=size(H,2)
                self.outputWeight = (eye(size(H,2))/self.regularizationParameter + H' * H) \ H' * outputData;
            else
                self.outputWeight = H' * ((eye(size(H,1))/self.regularizationParameter + H * H') \ outputData);
            end
            
            if ~(self.isLastLayer)
                A = self.outputWeight';
%                 B = A - mean(A,1);
                B = bsxfun(@(x,y) x-y,A,mean(A,1));
                C = cov(B);
                [Ve,Va] = eig(C);
                [~,I] = sort(diag(Va),'descend');
                Ve2 = Ve(:,I);
                self.pcaMatrix = Ve2(:,1:self.reducedDimension);
                projectedOutput = H*self.pcaMatrix;
            else
                self.pcaMatrix = [];
                projectedOutput = [];
            end
        end
        
        function H = hiddenLayerOutput(self, inputData, lastHiddenOutput)
            H = self.activationFunction(inputData*self.inputWeight + repmat(self.biasOfHiddenNeurons,[size(inputData,1),1]));
            if ~(self.isFirstLayer)
                H = [lastHiddenOutput, H];
            end
        end
        
        function out = predict(self, inputData, lastHiddenOutput)
            if (size(inputData,2) ~= self.numberOfInputNeurons)
                exception = MException('AESELMModule:inDim','Wrong input dimension!');
                throw(exception);
            end
            out = self.hiddenLayerOutput(inputData,lastHiddenOutput)*self.outputWeight;
        end
        
    end
end