classdef Accuracy < Metric
    methods (Static)
        function value = worstCase()
            value = 0;
        end
        function value = isRegressionMetric()
            value = false;
        end
        function value = calculate(pred,target)
            [~,argmax] = max(pred,[],2);
            [~,amax] = max(target,[],2);
            value = sum(argmax == amax)/length(amax);
        end
        function value = isBetter(val1, val2)
            value = val1 > val2;
        end
        
    end
end