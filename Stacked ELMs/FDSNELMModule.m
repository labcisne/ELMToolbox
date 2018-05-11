%   FDSNELMModule - Fast Deep Stacked Network Module
%   Train and Predict a Module of a FDSN
%
%   This code was implemented based on the following paper:
%
%   To be published...
%
%   This class should not be used outside the FDSNELM class.
%   This class is similar to the R-ELM, but uses the algorithm described in [1].
%   See the FDSNELM class help.

classdef FDSNELMModule < RELM
    
    properties
        numberOfOutputNeurons
        isFirstLayer
        totalNumberOfInputNeurons
    end
    
    methods (Access = ?FDSNELM)
        function self = FDSNELMModule(varargin)
            self = self@RELM(varargin{:});
        end
    end
    
    methods
        
        function [self,lastHiddenBeforeAct,layerOutput] = train(self,inputData,outputData,biasOfHiddenNeurons,lastHiddenBeforeAct,lastLayerOutput)
            if (~isempty(inputData) && size(inputData,2) ~= self.totalNumberOfInputNeurons)
                exception = MException('FDSNELMModule:wrongInputDimension','Wrong input dimension!');
                throw(exception);
            end
            
            auxTime = toc;
            if self.isFirstLayer
                lastHiddenBeforeAct = inputData*self.inputWeight + repmat(biasOfHiddenNeurons,[size(inputData,1),1]);
                self.biasOfHiddenNeurons = biasOfHiddenNeurons;
            else
                lastHiddenBeforeAct = lastHiddenBeforeAct + lastLayerOutput*self.inputWeight;
            end
            H = self.activationFunction(lastHiddenBeforeAct);
            
            if size(H,1)>=size(H,2)
                self.outputWeight = (eye(size(H,2))/self.regularizationParameter + H' * H) \ H' * outputData;
            else
                self.outputWeight = H' * ((eye(size(H,1))/self.regularizationParameter + H * H') \ outputData);
            end
            
            layerOutput = H*self.outputWeight;
            self.trainTime = toc - auxTime;
        end
        
        function [H,h] = hiddenLayerOutput(self, inputData, lastHiddenBeforeAct,lastLayerOutput)
            
            auxTime = toc;
            if self.isFirstLayer
                h = inputData*self.inputWeight + repmat(self.biasOfHiddenNeurons,[size(inputData,1),1]);
            else
                h = lastHiddenBeforeAct + lastLayerOutput*self.inputWeight;
            end
            H = self.activationFunction(h);
            self.lastTestTime = toc - auxTime;
            
        end
        
        function [out,lastHiddenBeforeAct] = predict(self, inputData, lastHiddenBeforeAct,lastLayerOutput)
            if (~isempty(inputData) && size(inputData,2) ~= self.totalNumberOfInputNeurons)
                exception = MException('FDSNELMModule:wrongInputDimension','Wrong input dimension!');
                throw(exception);
            end
            
            auxTime = toc;
            [out,lastHiddenBeforeAct] = self.hiddenLayerOutput(inputData, lastHiddenBeforeAct, lastLayerOutput);
            out = out*self.outputWeight;
            self.lastTestTime = toc - auxTime;
            
        end
    end
end