function [ cost ] = cost_cross_entropy( a, y )
    %COST_CROSS_ENTROPY 
    
    cost = -y .* log(a) - (1-y) .* log(1-a);
    cost(isinf(cost)) = flintmax(class(a) );    % Convert infinity to appropriate max float
end

