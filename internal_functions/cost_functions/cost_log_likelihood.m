function [ cost ] = cost_log_likelihood( a, y )
    %COST_MAXIMUM_LIKELIHOOD see: http://neuralnetworksanddeeplearning.com/chap3.html#softmax (80)
    
    assert( all(all(y==1|y==0)) );   % Note that this cost function works well for binary output encoding, it requires that labels be {1,0}
    
    cost = -log( sum(sum(y.*a)) );
end

