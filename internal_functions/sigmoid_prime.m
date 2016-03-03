function [ z_prime ] = sigmoid_prime( z )

    % Returns the derivative of the sigmoid function
    
    z_prime = sigmoid(z);
    z_prime = z_prime .* (1-z_prime);

end

