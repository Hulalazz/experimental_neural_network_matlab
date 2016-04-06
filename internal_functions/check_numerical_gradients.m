function check_numerical_gradients( model, grad_w, grad_b, X, y )
    %COMPUTE_NUMERICAL_GRADIENT Computes the numerical gradients for the cost function specified in the model parameters given the gradient of weights & biases, returns true/false on success or failure.

    epsilon = 1e-4;
    MAXDIFF = 1e-9;

%	sz = sum( [model.layer_sizes(1:end-1) .* model.layer_sizes(2:end) , model.layer_sizes(2:end)] );    % Compute the total size of all elements
    
    %
    % Compute numerical gradient, loop over all layers, biases, weights: perturb each weight, take the numericla gradient,
    % and compare it to the computed gradient.
    %
    %loss1 = zeros(sz,1);
    %loss2 = zeros(sz,1);
    for l = 1:model.num_layers-1                                    % Loop over each layer
        for j = 1:model.layer_sizes(l+1)                            % Loop over each node
            % Biases
            orig_bias = model.biases{l}(j);
            model.biases{l}(j) = orig_bias + epsilon;
            [a,~] = feedforward(model, X);
            cost1 = model.cost_function_cost( a{end}, y );
            model.biases{l}(j) = orig_bias - epsilon;
            [a,~] = feedforward(model, X);
            cost2 = model.cost_function_cost( a{end}, y );
            numerical_gradient = (double(cost1) - double(cost2)) / (2*epsilon);
            backprop_gradient  = grad_b{l}(j);
            %assert( abs(numerical_gradient - backprop_gradient) < MAXDIFF, 'Numerical gradient check failed' );
            model.biases{l}(j) = orig_bias;
            
            % Weights
            for k = 1:model.layer_sizes(l)                          % Loop over each input to each node
                jk = j*k+k;                                         % Index of edge j,k (edge from node j back to node k from the previous layer)
                switch model.update_method
                    case 'EG+-'
                        orig_weight = model.weights.positive{l}(jk);                    % Perturb the weight and calculate the cost
                        model.weights.positive{l}(jk) = orig_weight + epsilon;
                        [a,~] = feedforward(model, X);
                        cost1 = model.cost_function_cost( a{end}, y );
                        model.weights.positive{l}(jk) = orig_weight - epsilon;
                        [a,~] = feedforward(model, X);
                        cost2 = model.cost_function_cost( a{end}, y );
                        model.weights.positive{l}(jk) = origi_weight;
                    case 'GD'
                        orig_weight = model.weights{l}(jk);
                        model.weights{l}(jk) = orig_weight + epsilon;
                        [a,~] = feedforward(model, X);
                        cost1 = model.cost_function_cost( a{end}, y );
                        model.weights{l}(jk) = orig_weight - epsilon;
                        [a,~] = feedforward(model, X);
                        cost2 = model.cost_function_cost( a{end}, y );
                        model.weights{l}(jk) = orig_weight;
                    otherwise; error('Unrecognized update method');
                end
                numerical_gradient = (double(cost1) - double(cost2)) / (2*epsilon);             % Compare numerical gradient to the one obtained from backprop
                backprop_gradient  = grad_w{l}(jk);
                fprintf( 'backprop: [%f], numerical: [%f], diff: [%f]', backprop_gradient, numerical_gradient, abs(numerical_gradient - backprop_gradient) );
                %assert( abs(numerical_gradient - backprop_gradient) < MAXDIFF, 'Numerical gradient check failed' ); 
            end %for k (edge)
        end %for j (node)
    end %for l (layer)
    
end


%         orig_value = network_params_flat(i);        % Save the origianl value before perturbing the weight
%         
%         network_params_flat(i) = orig_value + epsilon;
%         [model.weights,model.biases] = unflatten_weights_biases( network_params_flat, model );
%         %[~,~,loss1(i)] = backprop( model, X, y );
%         
%         network_params_flat(i) = orig_value - epsilon;
%         [model.weights,model.biases] = unflatten_weights_biases( network_params_flat, model );
%         %[~,~,loss2(i)] = backprop( model, X, y );
%     end
%     
%     numerical_grad = (loss1 - loss2) / (2*epsilon);
%     
%     %derived_grad   = flatten_gradient_parameters( sz, grad_w, grad_b );
%     derived_grad   = flatten_weights_biases( model );
%     difference     = norm(numerical_grad-derived_grad) / norm(numerical_grad+derived_grad);
%     success_fail   = difference < MAXDIFF;

% % Adds epsilon to the models weight at the specified absolute index location. Index is relative to all weights and biases in the entire network, allowing for easy looping.
% function [ model ] = perturb( model, index, epsilon )
%     
%     % Generate an array of the sizes of the weights and biases, example, a network of size [784 30 30 10] will 
%     % come out to: [784*30 30*30 30*10 30 30 10], this will be used to identify the location of the index
%     network_layout = [ model.layer_sizes(1:end-1) .* model.layer_sizes(2:end) , model.layer_sizes(2:end) ];
%     cum_network = cumsum(network_layout);
%     position = find( cum_network >= index, 1 );
%     if (position > 1); local_ix = index - cum_network(position-1);
%     else               local_ix = index;    end;
%     
%     if(position <= model.num_layers-1)
%         % Update a weight
%         switch model.update_method
%             case 'EG+-'; model.weights.positive{position}(local_ix) = model.weights.positive{position}(local_ix) + epsilon;
%             case 'GD';   model.weights{position}(local_ix) = model.weights{position}(local_ix) + epsilon;
%         end
%     else
%         % Update a bias
%         model.biases{position-model.num_layers+1}(local_ix) = model.biases{position-model.num_layers+1}(local_ix) + epsilon;
%     end
% 
% end
% 
% function [ param_value ] = get_weight_or_bias_by_index( model, ix )
% 
% end
% 
% function update_weight_or_bias_by_index ( model, value )
% 
% end


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