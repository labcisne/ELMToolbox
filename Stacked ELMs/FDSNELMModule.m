%   FDSNELMModule - Fast Deep Stacked Network Module
%   Train and Predict a Module of a FDSN
%
%   This code was implemented based on the following paper:
%
%   [1] B. L. S. Silva, F. K. Inaba, P. M. Ciarelli (2018)
%       A fast algorithm to train a Deep Stacked Network using Extreme Learning Machine
%       Submitted to International Joint Conference on Neural Networks (IJCNN) 2018
%       Currently under review (06/02/18).
%
%   This class should not be used outside the FDSNELM class.
%   This class is similar to the R-ELM, but uses the algorithm described in [1].
%   See the FDSNELM class help.

classdef FDSNELMModule
    %Since the first module is different than the others, it is better not
    %to extend the R-ELM class, to avoid unwanted computations (it is not
    %necessary to generate a bias vector in all modules).
    properties (SetAccess = protected, GetAccess = public)
        numberOfHiddenNeurons
        numberOfInputNeurons
        numberOfOutputNeurons
        inputWeight
        biasOfHiddenNeurons = []
        regularizationParameter
        activationFunction
        outputWeight
        isFirstLayer
        newInputHidden = []
        seed
    end
    
    methods (Access = ?FDSNELM)
        
        function obj = FDSNELMModule(varargin)
%             if mod(nargin,2) ~= 0
%                 exception = MException('FDSNELMModule:ParameterError','Params must be given in pairs');
%                 throw (exception)
%             end
%             
%             for i=1:2:nargin
%                 if isprop(obj,varargin{i})
%                     obj.(varargin{i}) = varargin{i+1};
%                 else
%                     exception = MException('FDSNELMModule:ParameterError','Given parameter does not exist');
%                     throw (exception)
%                 end
%             end

            for i = 1:2:nargin
                obj.(varargin{i}) = varargin{i+1};
            end
            if isnumeric(obj.seed) && ~isempty(obj.seed)
                obj.seed = RandStream('mt19937ar','Seed', obj.seed);
            elseif ~isa(obj.seed, 'RandStream')
                obj.seed = RandStream.getGlobalStream();
            end
            if isempty(obj.numberOfInputNeurons)
                throw(MException('FDSNELMModule:emptyNumberOfInputNeurons','Empty Number of Input Neurons'));
            end

            if isequal(class(obj.activationFunction),'char')
                switch lower(obj.activationFunction)
                    case {'sig','sigmoid'}
                        %%%%%%%% Sigmoid
                        obj.activationFunction = @(tempH) 1 ./ (1 + exp(-tempH));
                    case {'sin','sine'}
                        %%%%%%%% Sine
                        obj.activationFunction = @(tempH) sin(tempH);
                    case {'hardlim'}
                        %%%%%%%% Hard Limit
                        obj.activationFunction = @(tempH) double(hardlim(tempH));
                    case {'tribas'}
                        %%%%%%%% Triangular basis function
                        obj.activationFunction = @(tempH) tribas(tempH);
                    case {'radbas'}
                        %%%%%%%% Radial basis function
                        obj.activationFunction = @(tempH) radbas(tempH);
                        %%%%%%%% More activation functions can be added here
                end
            elseif ~isequal(class(obj.activationFunction),'function_handle')
                exception = MException('FDSNELMModule:activationFunctionError','Hidden activation function not supported');
                throw (exception)
            end
            
            if ~obj.isFirstLayer && isempty(obj.newInputHidden)
                obj.newInputHidden = -1 + 2*rand(obj.seed,obj.numberOfOutputNeurons,obj.numberOfHiddenNeurons);
            end
            
        end
        
        function [self,lastHiddenBeforeAct,layerOutput] = train(self,inputData,outputData,biasOfHiddenNeurons,lastHiddenBeforeAct,lastLayerOutput)
            
%             if (self.isFirstLayer && size(inputData,2) ~= self.numberOfInputNeurons)
%                 exception = MException('FDSNELMModule:wrongNumberOfInputNeurons','Wrong input dimension!');
%                 throw(exception);
%             end
            
            if self.isFirstLayer
                lastHiddenBeforeAct = inputData*self.inputWeight + repmat(biasOfHiddenNeurons,[size(inputData,1),1]);
                self.biasOfHiddenNeurons = biasOfHiddenNeurons;
            else
                lastHiddenBeforeAct = lastHiddenBeforeAct + lastLayerOutput*self.newInputHidden;
            end
            H = self.activationFunction(lastHiddenBeforeAct);
            
            if size(H,1)>=size(H,2)
                self.outputWeight = (eye(size(H,2))/self.regularizationParameter + H' * H) \ H' * outputData;
            else
                self.outputWeight = H' * ((eye(size(H,1))/self.regularizationParameter + H * H') \ outputData);
            end
            
            layerOutput = H*self.outputWeight;
            
        end
        
        function [H,h] = hiddenLayerOutput(self, inputData, lastHiddenBeforeAct,lastLayerOutput)
            
            if self.isFirstLayer
                h = inputData*self.inputWeight + repmat(self.biasOfHiddenNeurons,[size(inputData,1),1]);
            else
                h = lastHiddenBeforeAct + lastLayerOutput*self.newInputHidden;
            end
            H = self.activationFunction(h);
            
        end
        
        function [out,lastHiddenBeforeAct] = predict(self, inputData, lastHiddenBeforeAct,lastLayerOutput)
            if (~isempty(inputData) && size(inputData,2) ~= self.numberOfInputNeurons)
                exception = MException('FDSNELMModule:wrongInputDimension','Wrong input dimension!');
                throw(exception);
            end
            
            [out,lastHiddenBeforeAct] = self.hiddenLayerOutput(inputData, lastHiddenBeforeAct, lastLayerOutput);
            out = out*self.outputWeight;
            
        end
    end
end