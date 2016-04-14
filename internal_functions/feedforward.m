function [ a, z ] = feedforward( model, input )
    % Input model (weights, biases, etc), and X's
    % Return the activations (post sigmoid), and z values (pre-sigmoid x*w calculations)
    
    if(isa(model,'cell')); model = cell2mat(model); end
    
    a = cell(model.num_layers, 1);  % Hold activations
	z = cell(model.num_layers, 1);  % Hold weight inputs, z values at each later
    
    a{1} = input;
    
    % Perform JL projections if they are enabled in general, and defined for the first layer (input layer)
    if( isfield(model, 'jl_k_layer_sizes') && model.jl_k_layer_sizes(1) ~= -1 )
        a{1} = (a{1} * model.jl_projection_matricies{1}) ./ sqrt(model.jl_k_layer_sizes(1));
    end
    
    % Compute activations and z for each layer
	for l = 2:(model.num_layers)
        [weights, biases] = return_weights_biases_from_model(model, l-1);
        z{l} = a{l-1} * weights + repmat( biases, [size(a{l-1},1) 1] ); % TODO make this bias addition more efficient with bsxfun or other, but check it in the profiler first.
        
        if( use_softmax_at_this_layer(model, l) )       %TODO this is a work in progress and probably not as clean as it needs to ultimately be
            a{l} = softmax_function( z{l} );
        else
            a{l} = sigmoid( z{l} );
        end
        
        % Perform JL projections if: (1) they're enabled, (2) they're enabled for this layer (e.g. not set to -1)
        if( isfield(model, 'jl_k_layer_sizes') && model.jl_k_layer_sizes(l) ~= -1 )
            %z{l} = z{l} * model.jl_projection_matricies{l} ./ sqrt(model.jl_k_layer_sizes(l));
            a{l} = a{l} * model.jl_projection_matricies{l} ./ sqrt(model.jl_k_layer_sizes(l));
        end
	end

end


% This function checks whether we are using EG+- or GD, if it's EG+- it combines the pos & neg weights and returns them, 
% else just returns the normal weights used by GD.
% This function is just here to clean up the case statements and organize them in one place for clarity.
function [weights, biases] = return_weights_biases_from_model(model, layer)
    switch( model.update_method )
        case 'EG+-'
        	weights = (model.weights.positive{layer} - model.weights.negative{layer});
        case 'GD'
            weights = model.weights{layer};
        otherwise
            assert( false, 'Update method not recognized.' );
    end
    biases = model.biases{layer};
end


% This function determines if the given layer should use a softmax output or not.
% Current implementation only checks the last layer, but it's coded this way for easy extensibility
% and general code readability.
function [truefalse] = use_softmax_at_this_layer(model, layer)
    truefalse = model.use_softmax_output_layer && layer == model.num_layers;
end
