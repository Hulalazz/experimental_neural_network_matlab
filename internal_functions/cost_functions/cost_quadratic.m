function [ cost ] = cost_quadratic( a, y )
    %COST_QUADRATIC Quadratic cost function

    cost = (1 / (2*size(a,1))) * sum(sqrt(sum((a-y).^2,2)).^2);

end
