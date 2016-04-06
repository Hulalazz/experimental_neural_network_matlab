function [ delta ] = cost_quadratic_delta( z, a, y )
    %COST_QUADRATIC_DELTA Summary of this function goes here
    % Remember that the cost_delta function needs to return the delta w.r.t. all mini batch samples, 
    % we can't average it here unless we also compute the gradient w.r.t. weights here since that computation needs
    % all samples.
    
    delta = (a - y) .* sigmoid_prime(z);

end

