classdef aRRMSE < RRMSE
    methods (Static)
        function value = calculate(pred,target)
            value = mean(RRMSE.calculate(pred,target));
        end
    end
end