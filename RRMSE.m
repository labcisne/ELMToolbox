classdef RRMSE < Metric
    methods (Static)
        function value = worstCase()
            value = Inf;
        end
        function value = isRegressionMetric()
            value = true;
        end
        function value = calculate(pred,target)
            value = sqrt(sum((target - pred).^2,1)./sum((mean(target,1)-target).^2,1));
        end
        function value = isBetter(val1, val2)
            value = val1 < val2;
        end
    end
end