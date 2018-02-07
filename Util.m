classdef (Abstract) Util
    properties (SetAccess = protected, GetAccess = public)
        trainTime
        lastTestTime
        trainPartialMetric
        testPartialMetric
        seed = []
    end
    methods
        
        function acc = calculateAccuracy(~,pred,target)
            [~,argmax] = max(pred,[],2);
            [~,amax] = max(target,[],2);
            acc = sum(argmax == amax)/length(amax);
        end
        
        function rmse = calculateRMSE(~,pred, target)
            rmse = sqrt(mean(mean((target - pred).^2)));
        end
        
        function projMatrix = PCA(~,A, numEigs)
            %             A = matrix;
            %             B = A - mean(A,1);
            B = bsxfun(@(x,y) x-y,A,mean(A,1));
            C = cov(B);
            [Ve,Va] = eig(C);
            [~,I] = sort(diag(Va),'descend');
            Ve2 = Ve(:,I);
            projMatrix = Ve2(:,1:numEigs);
        end
        
        function actFun = parseActivationFunction(~,actFun)
            if isequal(class(actFun),'char')
                switch lower(actFun)
                    case {'sig','sigmoid'}
                        %%%%%%%% Sigmoid
                        actFun = @(tempH) 1 ./ (1 + exp(-tempH));
                    case {'sin','sine'}
                        %%%%%%%% Sine
                        actFun = @(tempH) sin(tempH);
                    case {'hardlim'}
                        %%%%%%%% Hard Limit
                        actFun = @(tempH) double(hardlim(tempH));
                    case {'tribas'}
                        %%%%%%%% Triangular basis function
                        actFun = @(tempH) tribas(tempH);
                    case {'radbas'}
                        %%%%%%%% Radial basis function
                        actFun = @(tempH) radbas(tempH);
                        %%%%%%%% More activation functions can be added here
                end
            elseif ~isequal(class(actFun),'function_handle')
                throw(MException('Util:activationFunctionError','Error Activation Function'));
            end
        end
        
        function seed = parseSeed(self)
            if isnumeric(self.seed) && ~isempty(self.seed)
                seed = RandStream('mt19937ar','Seed', self.seed);
            elseif ~isa(self.seed, 'RandStream')
                seed = RandStream.getGlobalStream();
            else
                seed = self.seed;
            end
        end
        
    end
    methods (Abstract)
        self = train(X,Y);
        pred = predict(X);
    end
end