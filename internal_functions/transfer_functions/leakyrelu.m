function [ a ] = leakyrelu( z )
    %LEAKYRELU Summary of this function goes here

    a = z .* ( ((z<=0).*(-0.99))+1 );

end

