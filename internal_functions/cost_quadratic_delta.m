function [ delta ] = cost_quadratic_delta( z, a, y )
    %COST_QUADRATIC_DELTA Summary of this function goes here
    
    delta = (a - y) .* sigmoid_prime(z);

end

