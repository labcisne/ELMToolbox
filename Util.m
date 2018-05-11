classdef (Abstract) Util < handle
    properties (SetAccess = protected, GetAccess = public)
        trainTime
        lastTestTime
        trainPartialMetric
        testPartialMetric
        seed = []
    end
    methods
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
    methods (Static)
        function [dta, param] = normalizeData(dta, param) 
            if nargin == 1
                mindta = min(dta);
                maxdta = max(dta);
                maxdta(maxdta == mindta) = maxdta(maxdta == mindta) + eps(maxdta(maxdta == mindta));
                param = [mindta; maxdta];
            else
                mindta = param(1,:);
                maxdta = param(2,:);
            end
            dta = -1 + 2*(dta - mindta)./(maxdta - mindta);
        end
        function dta = unNormalizeData(dta, param)
            mindta = param(1,:);
            maxdta = param(2,:);
            dta = ((dta+1)/2).*(maxdta - mindta) + mindta;
        end
        function projMatrix = PCA(A, numEigs)
            %             A = matrix;
            %                 B = A - mean(A,1);
            B = bsxfun(@(x,y) x-y,A,mean(A,1));
            C = cov(B);
            [Ve,Va] = eig(C);
            [~,I] = sort(diag(Va),'descend');
            Ve2 = Ve(:,I);
            projMatrix = Ve2(:,1:numEigs);
        end
    end
end