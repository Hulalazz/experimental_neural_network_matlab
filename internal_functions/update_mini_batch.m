function [ model ] = update_mini_batch( model, mini_batch_X, mini_batch_y )
    m = size(mini_batch_X,1);

    % delta_w, and delta_b are the gradient for weights and biases w.r.t. the cost function
    % They are averaged across all samples in the mini_batch by backprop
	[delta_w, delta_b, cost] = backprop( model, mini_batch_X, mini_batch_y );
    
    % Check numerical gradients if debug option for doing so is enabled
    % TODO bug concern: need to move the averaging over minibatch inside the cost function right??!!??
    if( model.debug_check_numerical_gradients );  assert( check_numerical_gradients(model, delta_w, delta_b, mini_batch_X, mini_batch_y), 'Numerical gradient check failed.' );  end

    % Metrics - save the training cost in the Metrics data
    if(model.monitor_training_cost); model.Metrics.training_cost{end+1} = cost/m; end;

    % Update weights and biases
    for l = 1:model.num_layers-1
        switch( model.update_method )
            case 'EG+-'
                rpos = exp(-model.learning_rate .* delta_w{l});
                rneg = 1 ./ rpos;
                weights_pos_update = rpos .* model.weights.positive{l};
                weights_neg_update = rneg .* model.weights.negative{l};
                if( strcmp(model.U, 'unnormalized') )
                    model.weights.positive{l} = weights_pos_update;
                    model.weights.negative{l} = weights_neg_update;
                else
                    sum_inputs_to_neuron = sum(weights_pos_update,1) + sum(weights_neg_update,1);
                    model.weights.positive{l} = model.U .* bsxfun(@rdivide, weights_pos_update, sum_inputs_to_neuron);
                    model.weights.negative{l} = model.U .* bsxfun(@rdivide, weights_neg_update, sum_inputs_to_neuron);
                end
            case 'GD'
                model.weights{l} = regularized_weights(model.lambda, model.learning_rate, model.weights{l}, model.regularization) - (model.learning_rate .* delta_w{l});    % Original formulation without regularization:  model.weights{l} = model.weights{l} - (learning_rate .* delta_w{l});
            otherwise
                assert( false, 'Update method not recognized.' );
        end
        model.biases{l}  = model.biases{l}  - (model.learning_rate .* delta_b{l});
    end

end

% Return (1 - regularization) * weights. 
%
% This function is effectively the same as regularization_term, it just applies the regularization in one step to the weights 
%   rather than processing it as two steps. It's simply more efficient in terms of computation
% It returns the regularized weights rather than the regularization term its self. For example gradient descent can apply 
%   regularization in this more efficient way, but EG+- updates need the regularization term separate.
% Note: It might not be worth the computational efficiency of separating these two, so they may collapse into one function in the future.
function [reg_weights] = regularized_weights(lambda, eta, weights, regularization_method)
    switch regularization_method
        case 'L2'
            reg_weights = (1 - eta*lambda) .* weights;
        case 'L1'
            reg_weights = weights - ( (eta*lambda) .* sign(weights) );
        case 'none'
            reg_weights = weights;
        otherwise
            assert( false, sprintf('Regularization method [%s] not recognized', regularization_method) );
    end
end

