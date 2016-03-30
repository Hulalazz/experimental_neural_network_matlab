function [ delta ] = cost_log_likelihood_delta( z, a, y )
    %COST_LOG_LIKELIHOOD_DELTA
    % http://neuralnetworksanddeeplearning.com/chap3.html#softmax (82)

    assert( all(all(y==1|y==0)) );   % Note that this cost function works well for binary output encoding, it requires that labels be {1,0}
    
    delta = a - y;
    %delta = -(a .* y);

end

