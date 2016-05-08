function [ model ] = update_mini_batch( model, mini_batch_X, mini_batch_y )

    dropout;    % Set up random dropout vectors if dropout is enabled
    
    % delta_w, and delta_b are the gradient for weights and biases w.r.t. the cost function
    % They are averaged across all samples in the mini_batch by backprop
	[delta_w, delta_b, cost] = backprop( model, mini_batch_X, mini_batch_y );
    
    % Check numerical gradients if debug option for doing so is enabled
    % TODO bug concern: need to move the averaging over minibatch inside the cost function right??!!??
    if( model.debug_check_numerical_gradients );  check_numerical_gradients(model, delta_w, delta_b, mini_batch_X, mini_batch_y);  end

    % Metrics - 1) save the training cost in the Metrics data, 2) save length of delta_b
    if(model.monitor_training_cost); model.Metrics.training_cost{end+1} = cost; end;
    if(model.monitor_delta_norm)
        delta_norm = zeros( model.num_layers-1, 1 );
        for ix = 1:numel(delta_b)
            delta_norm(ix) = norm(delta_b{ix});
        end
        model.Metrics.delta_norm{end+1} = delta_norm;
    end

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
                    eg_sharing_regularization         % Conditional EG fixed regularization
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

    % Optional save weight & bias history
    if(model.store_weight_history); model.Metrics.weight_bias_history{end+1} = flatten_weights_biases( model, true ); end;
    
    %
    %
    % Nested functions
    %
    %
    function dropout
        if( ~isempty(model.dropout_p) )
            for i = 1:model.num_layers
                model.Scratch.dropout_r{i} = binornd( 1, model.dropout_p(i), [model.mini_batch_size model.layer_sizes(i)] );     %% TODO: Speed improvement - layers that have a 0 dropout don't need a full matrix here, the cell can just be a scalar 1 value, simple if statement here should be enough.
            end
        end
    end

    % Adds a fixed (alpha) regularization term to all weights before normalization.
    function eg_sharing_regularization
        switch(model.EG_sharing)
            case 'fixed_regularization'
                weights_pos_update = weights_pos_update + model.EG_sharing_alpha;
                weights_neg_update = weights_neg_update + model.EG_sharing_alpha;
            case 'past_average'
                n = double(model.Scratch.batch_num);
                alpha = model.EG_sharing_alpha;
                
                % Init past average data structure first iteration
                if( n == 1 )
                    if( ~isfield(model.Scratch, 'EG_sharing_weights') )
                        model.Scratch.EG_sharing_weights = struct;
                        model.Scratch.EG_sharing_weights.positive = cell( size(model.weights.positive) );
                        model.Scratch.EG_sharing_weights.negative = cell( size(model.weights.negative) );
                    end
                    model.Scratch.EG_sharing_weights.positive{l} = zeros( size(model.weights.positive{l}) );
                    model.Scratch.EG_sharing_weights.negative{l} = zeros( size(model.weights.negative{l}) );
                end
                
                % Store average of previous weights
                model.Scratch.EG_sharing_weights.positive{l} = (model.Scratch.EG_sharing_weights.positive{l}.*(n-1) + model.weights.positive{l}) ./ n;
                model.Scratch.EG_sharing_weights.negative{l} = (model.Scratch.EG_sharing_weights.negative{l}.*(n-1) + model.weights.negative{l}) ./ n;
                
                % Update weights with sharing
                weights_pos_update = ((1-alpha) .* weights_pos_update) + (alpha .* model.Scratch.EG_sharing_weights.positive{l});
                weights_neg_update = ((1-alpha) .* weights_neg_update) + (alpha .* model.Scratch.EG_sharing_weights.negative{l});
                
            case 'none' % Explicitly do nothing
            otherwise; error('Code unreachable, bug otherwise.');
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

