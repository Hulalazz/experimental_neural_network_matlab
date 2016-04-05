function [ success_fail ] = check_numerical_gradients( model, grad_w, grad_b, X, y )
    %COMPUTE_NUMERICAL_GRADIENT Computes the numerical gradients for the cost function specified in the model parameters given the gradient of weights & biases, returns true/false on success or failure.

    epsilon = 1e-4;
    MAXDIFF = 1e-9;

	sz = sum( [model.layer_sizes(1:end-1) .* model.layer_sizes(2:end) , model.layer_sizes(2:end)] );    % Compute the total size of all elements
    
    %
    % Compute numerical gradient, perturb each weight and calculate the cost
    %
    loss1 = zeros(sz,1);
    loss2 = zeros(sz,1);
    
    for i = 1:sz
        [~,~,loss1(i)] = backprop( perturb(model, i, epsilon), X, y );
        [~,~,loss2(i)] = backprop( perturb(model, i, -epsilon), X, y );
    end
    
    derived_grad   = flatten_gradient_parameters( sz, grad_w, grad_b );
    numerical_grad = (loss1 - loss2) / (2*epsilon);
    difference     = norm(numerical_grad-derived_grad) / norm(numerical_grad+derived_grad);
    success_fail   = difference < MAXDIFF;
end

% Adds epsilon to the models weight at the specified absolute index location. Index is relative to all weights and biases in the entire network, allowing for easy looping.
function [ model ] = perturb( model, index, epsilon )
    
    % Generate an array of the sizes of the weights and biases, example, a network of size [784 30 30 10] will 
    % come out to: [784*30 30*30 30*10 30 30 10], this will be used to identify the location of the index
    network_layout = [ model.layer_sizes(1:end-1) .* model.layer_sizes(2:end) , model.layer_sizes(2:end) ];
    cum_network = cumsum(network_layout);
    position = find( cum_network >= index, 1 );
    if (position > 1); local_ix = index - cum_network(position-1);
    else               local_ix = index;    end;
    
    if(position <= model.num_layers-1)
        % Update a weight
        switch model.update_method
            case 'EG+-'; model.weights.positive{position}(local_ix) = model.weights.positive{position}(local_ix) + epsilon;
            case 'GD';   model.weights{position}(local_ix) = model.weights{position}(local_ix) + epsilon;
        end
    else
        % Update a bias
        model.biases{position-model.num_layers+1}(local_ix) = model.biases{position-model.num_layers+1}(local_ix) + epsilon;
    end

end

% TODO swap out with global functions flatten_weights_biases & unflatten_weights_biases
function [ flattened_params ] = flatten_gradient_parameters( sz, grad_w, grad_b )
    flattened_params = zeros(sz,1);
    pos = 1;
    for i = 1:numel(grad_w)
        flattened_params(pos:pos+numel(grad_w{i})-1) = grad_w{i}(:);
        pos = pos + numel(grad_w{i});
    end
    for i = 1:numel(grad_b)
        flattened_params(pos:pos+numel(grad_b{i})-1) = grad_b{i}(:);
        pos = pos + numel(grad_b{i});
    end
        
end