function [ weights_biases_flat ] = flatten_weights_biases( model, varargin )
    %FLATTEN_WEIGHTS_BIASES Flattens the weights and biases into a single column vector
    
    array_size = num_network_parameters( model );
    weights_biases_flat = zeros( array_size, 1, 'like', model.biases{1} );
    p = 1;
    
    for i = 1:model.num_layers-1
        % Weights
        switch model.update_method
            case 'EG+-'
                n = numel( model.weights.positive{i} );
                weights_biases_flat(     (p):(p+n-1)   ) = model.weights.positive{i}(:);
                weights_biases_flat(   (p+n):(p+2*n-1) ) = model.weights.negative{i}(:);
                p = p + 2*n;
            case 'GD'
                n = numel( model.weights{i} );
                weights_biases_flat( (p):(p+n-1) ) = model.weights{i}(:);
                p = p + n;
            otherwise; error( 'update_method not recognized' );
        end
        % Biases
        n = numel( model.biases{i} );
        weights_biases_flat( (p):(p+n-1) ) = model.biases{i};
        p = p + n;
    end

end

