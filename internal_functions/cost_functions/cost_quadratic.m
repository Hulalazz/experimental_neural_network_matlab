function [ cost ] = cost_quadratic( a, y )
    %COST_QUADRATIC Quadratic cost function

    cost =  0.5 * norm(a-y)^2;

end
