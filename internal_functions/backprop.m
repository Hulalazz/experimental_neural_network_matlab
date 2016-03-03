function [ grad_w, grad_b, optional_cost ] = backprop( model, X, y )
    num_layers = model.num_layers;
    m = size(X,1);                          % mini batch size
    
    grad_b      = cell(num_layers-1, 1);    % Change in Cost with respect to biases
    grad_w      = cell(num_layers-1, 1);    % Change in Cost with respect to weights
    
    % feedforward
    [ activations, z ] = feedforward( model, X );

    % Backward pass
    % output layer
    delta = model.cost_function_delta( z{end}, activations{end}, y );
    grad_b{end} = mean( delta, 1 );
    grad_w{end} = (activations{end-1}' * delta) ./ m;

    % all other layers
    for l = (num_layers-2) : -1 : 1
        switch model.update_method;
            case 'EG+-'; weights_Lplus1 = model.weights.positive{l+1} - model.weights.negative{l+1};
            case 'GD';   weights_Lplus1 = model.weights{l+1};
            otherwise;   assert(false,'Update method not recognized');
        end
        %delta = ( delta * model.weights{l+1}' ) .* sigmoid_prime(z{l+1});
        delta = ( delta * weights_Lplus1' ) .* sigmoid_prime(z{l+1});
        grad_b{l} = mean( delta, 1 );
        grad_w{l} = activations{l}' * delta ./ m;
    end

%TODO: 
    % Generate the sum of squares cost function output if it was requested
    if( nargout == 3 )
        %optional_cost = sum(sum( (1/2) * ((y - activations{end}) .^ 2) ) );
        optional_cost = model.cost_function_cost( activations{end}, y );
    end
end

