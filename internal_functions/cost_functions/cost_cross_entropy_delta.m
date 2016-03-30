function [ delta ] = cost_cross_entropy_delta( z, a, y )
    %COST_CROSS_ENTROPY_DELTA 

    delta = a - y;
    
end

