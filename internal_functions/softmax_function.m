function [ softmax_of_z ] = softmax_function( z )
    %SOFTMAX Compute the softmax function for a given layer

    softmax_of_z = bsxfun( @rdivide, exp(z), sum(exp(z),2) );

end

