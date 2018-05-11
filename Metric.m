classdef Metric < handle
    methods (Abstract,Static)
        value = calculate(pred,target);       
        value = worstCase();
        value = isRegressionMetric();
        value = isBetter(val1, val2);
    end
end