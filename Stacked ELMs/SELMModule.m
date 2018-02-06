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

classdef SELMModule
    %Since the first module is different than the others, it is better not
    %to extend the R-ELM class, to avoid unwanted computations (the
    %varargin variable should be checked every time to separate the
    %properties that will be used to construct the R-ELM object, and
    %check if a module is the first or last, so the correct matrices
    %are generated and stored).
    properties (SetAccess = protected, GetAccess = public)
        isFirstLayer
        isLastLayer
        numberOfHiddenNeurons
        reducedDimension
        numberOfInputNeurons
        inputWeight
        biasOfHiddenNeurons
        regularizationParameter
        activationFunction
        outputWeight
        pcaMatrix
        seed
    end
    
    methods (Access = {?SELM,?AESELMModule})
        function obj = SELMModule(varargin)
            
            
%             if mod(nargin,2) ~= 0
%                 exception = MException('SELMModule:ParameterError','Params must be given in pairs');
%                 throw (exception)
%             end
%             
%             for i=1:2:nargin
%                 if isprop(obj,varargin{i})
%                     obj.(varargin{i}) = varargin{i+1};
%                 else
%                     exception = MException('SELMModule:ParameterError','Given parameter does not exist');
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
                throw(MException('SELMModule:emptyNumberOfInputNeurons','Empty Number of Input Neurons'));
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
                exception = MException('SELMModule:activationFunctionError','Hidden activation function not supported');
                throw (exception)
            end
            
            if obj.isFirstLayer
                obj.inputWeight = -1 + 2*rand(obj.seed,obj.numberOfInputNeurons,obj.numberOfHiddenNeurons);
                obj.biasOfHiddenNeurons = rand(obj.seed,1,obj.numberOfHiddenNeurons);
            else
                obj.inputWeight = -1 + 2*rand(obj.seed,obj.numberOfInputNeurons,obj.numberOfHiddenNeurons-obj.reducedDimension);
                obj.biasOfHiddenNeurons = rand(obj.seed,1,obj.numberOfHiddenNeurons-obj.reducedDimension);
            end
            
        end
    end
    methods
        function [self, projectedOutput] = train(self,inputData,outputData,lastHiddenOutput)
            
%             if (size(inputData,2) ~= self.numberOfInputNeurons)
%                 exception = MException('SELMModule:wrongNumberOfInputNeurons','Wrong input dimension!');
%                 throw(exception);
%             end
            
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
                A = self.outputWeight';
%                 B = A - mean(A,1);
                B = bsxfun(@(x,y) x-y,A,mean(A,1));
                C = cov(B);
                [Ve,Va] = eig(C);
                [~,I] = sort(diag(Va),'descend');
                Ve2 = Ve(:,I);
                self.pcaMatrix = Ve2(:,1:self.reducedDimension);
                projectedOutput = H*self.pcaMatrix;
                %                 self.outputWeight = []; %Uncomment this if you want to see how a metric 'evolves' over time
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
%             if (size(inputData,2) ~= self.numberOfInputNeurons)
%                 exception = MException('SELMModule:wrongInputDimension','Wrong input dimension!');
%                 throw(exception);
%             end
            
            out = self.hiddenLayerOutput(inputData,lastHiddenOutput)*self.outputWeight;
            
        end
        
        
    end
    
end