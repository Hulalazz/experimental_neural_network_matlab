function [ lrp ] = leakyrelu_prime( z )
    %LEAKYRELU_PRIME Summary of this function goes here

    lrp = ((z<=0).*(-0.99))+1;

end

