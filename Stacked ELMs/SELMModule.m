%   SELMModule - Stacked Extreme Learning Machine Module Class
%   Train and Predict a Module of a Stacked Extreme Learning Machine
%
%   This code was implemented based on the following paper:
%
%   [1] Zhou, H., Huang, G.-B., Lin, Z., Wang, H., & Soh, Y. C. (2014).
%       Stacked Extreme Learning Machines.
%       IEEE Transactions on Cybernetics, PP(99), 1.
%       https://doi.org/10.1109/TCYB.2014.2363492
%       (http://ieeexplore.ieee.org/document/6937189/)
%
%   This class should not be used outside the SELM class.
%   This class is similar to the R-ELM, but uses the algorithm described in [1].
%   See the SELM class help.

classdef SELMModule < RELM
    
    properties
        isFirstLayer
        isLastLayer
        totalNumberOfHiddenNeurons
        reducedDimension
        pcaMatrix
    end
    
    methods (Access = {?SELM,?AESELMModule})
        function self = SELMModule(varargin)
            self = self@RELM(varargin{:});
        end
    end
    
    methods
        function [self, projectedOutput] = train(self,inputData,outputData,lastHiddenOutput)
            
            auxTime = toc;
            H = self.activationFunction(inputData*self.inputWeight + repmat(self.biasOfHiddenNeurons,[size(inputData,1),1]));
            
            if ~(self.isFirstLayer)
                H = [lastHiddenOutput, H];
            end
            
            if size(H,1)>=size(H,2)
                self.outputWeight = (eye(size(H,2))/self.regularizationParameter + H' * H) \ H' * outputData;
            else
                self.outputWeight = H' * ((eye(size(H,1))/self.regularizationParameter + H * H') \ outputData);
            end
            
            if ~(self.isLastLayer)
                self.pcaMatrix = self.PCA(self.outputWeight',self.reducedDimension);
                projectedOutput = H*self.pcaMatrix;
                %                 self.outputWeight = []; %Uncomment this if you want to see how a metric 'evolves' over time
            else
                self.pcaMatrix = [];
                projectedOutput = [];
            end
            self.trainTime = toc - auxTime;
            
        end
        
        function H = hiddenLayerOutput(self, inputData, lastHiddenOutput)
            H = self.activationFunction(inputData*self.inputWeight + repmat(self.biasOfHiddenNeurons,[size(inputData,1),1]));
            if ~(self.isFirstLayer)
                H = [lastHiddenOutput, H];
            end
        end
        
        function out = predict(self, inputData, lastHiddenOutput)
            auxTime = toc;
            out = self.hiddenLayerOutput(inputData,lastHiddenOutput)*self.outputWeight;
            self.lastTestTime = toc - auxTime;
        end
        
    end
end