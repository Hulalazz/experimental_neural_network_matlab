function [ model ] = update_mini_batch( model, mini_batch_X, mini_batch_y, learning_rate )
    m = size(mini_batch_X,1);

    % Initialize data structure to sum the weights and biases for averaging
%    sum_w = cell(model.num_layers-1,1);
%    sum_b = cell(model.num_layers-1,1);
%    for l = 1:model.num_layers-1
%        sum_w{l} = zeros(size(model.weights{l}));
%        sum_b{l} = zeros(size(model.biases{l}));
%    end

    % delta_w, and delta_b are the gradient for weights and biases w.r.t. the cost function
    % They are averaged across all samples in the mini_batch by backprop
	[delta_w, delta_b, cost] = backprop( model, mini_batch_X, mini_batch_y );
    
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
                sum_inputs_to_neuron = sum(weights_pos_update,1) + sum(weights_neg_update,1);
                EG_reg_term = regularization_term(model.lambda, model.learning_rate, model.weights.positive{l}-model.weights.negative{l}, model.regularization);
                model.weights.positive{l} = model.U .* bsxfun(@rdivide, weights_pos_update, sum_inputs_to_neuron) - EG_reg_term;
                model.weights.negative{l} = model.U .* bsxfun(@rdivide, weights_neg_update, sum_inputs_to_neuron) - EG_reg_term;
            case 'GD'
                model.weights{l} = regularized_weights(model.lambda, model.learning_rate, model.weights{l}, model.regularization) - (learning_rate .* delta_w{l});    % Original formulation without regularization:  model.weights{l} = model.weights{l} - (learning_rate .* delta_w{l});
                model.biases{l}  = model.biases{l}  - (learning_rate .* delta_b{l});
            otherwise
                assert( false, 'Update method not recognized.' );
        end
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

% Returns the regularization term for the various regularization methods and appropriate parameters.
function [reg_term] = regularization_term(lambda, eta, weights, regularization_method)
    switch regularization_method
        case 'L2'
            reg_term = (eta*lambda) .* weights;
        case 'L1'
            reg_term = ( (eta*lambda) .* sign(weights) );
        case 'none'
            reg_term = zeros(size(weights));
        otherwise
            assert( false, sprintf('Regularization method [%s] not recognized', regularization_method) );
    end
end