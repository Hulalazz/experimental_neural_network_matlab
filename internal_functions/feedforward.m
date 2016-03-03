function [ a, z ] = feedforward( model, input )
    % Input model (weights, biases, etc), and X's
    % Return the activations (post sigmoid), and z values (pre-sigmoid x*w calculations)
    
    a = cell(model.num_layers, 1);  % Hold activations
	z = cell(model.num_layers, 1);  % Hold weight inputs, z values at each later
    
    a{1} = input;

    switch( model.update_method )
        case 'EG+-'
            for l = 1:(model.num_layers-1)
                z{l+1} = a{l} * (model.weights.positive{l} - model.weights.negative{l}) + repmat( model.biases{l}, [size(a{l},1) 1] );
                a{l+1} = sigmoid( z{l+1} );
            end
        case 'GD'        
            for l = 1:(model.num_layers-1)
                z{l+1} = a{l} * model.weights{l} + repmat( model.biases{l}, [size(a{l},1) 1] ); % TODO make this bias addition more efficient with bsxfun or other, but check it in the profiler first.
                a{l+1} = sigmoid( z{l+1} );
            end
        otherwise
            assert( false, 'Update method not recognized.' );
    end
end

