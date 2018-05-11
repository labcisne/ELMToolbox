classdef RMSE < Metric
    methods (Static)
        function value = worstCase()
            value = Inf;
        end
        function value = isRegressionMetric()
            value = true;
        end
        function value = calculate(pred,target)
            value = sqrt(mean(mean((target - pred).^2)));
        end
        function value = isBetter(val1, val2)
            value = val1 < val2;
        end
    end
end