function [ cost ] = cost_cross_entropy( a, y )
    %COST_CROSS_ENTROPY 
    % Note this returns total cost, not averaged over samples, that must be done outside this function.
    
    cost = -y .* log(a) - (1-y) .* log(1-a);
    cost(isnan(cost)) = 0;                      % Convert nan to 0 for casees such as when a and y are both 1, this should be 0 (1-y=0 in this case)
    cost(isinf(cost)) = flintmax(class(a));     % Convert infinity to appropriate max float
    cost = sum(cost(:))  / size(a,1);
end

