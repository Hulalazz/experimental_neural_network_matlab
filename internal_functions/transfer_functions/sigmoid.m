function [ sigma ] = sigmoid( z )
    sigma = 1 ./ (1 + exp(-z) );
end

