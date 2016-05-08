function [ weights_biases_flat ] = flatten_weights_biases( model, optional_include_biases )
    %FLATTEN_WEIGHTS_BIASES Flattens the weights and biases into a single column vector

% 	p = inputParser;
%     addRequired( p, 'model' );
%     addOptional( p, 'include_biases', true );
%     parse( p, model, varargin{:} );

    % Optional include biases
    if(nargin > 1); include_biases = optional_include_biases;
    else include_biases = true; end;
    
    if( strcmp( model.update_method, 'EG+-' ) ); array_size = num_network_parameters( model, model.EG_sharing_inc_biases );
    else array_size = num_network_parameters( model ); end
    
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
        if( include_biases )
            n = numel( model.biases{i} );
            weights_biases_flat( (p):(p+n-1) ) = model.biases{i};
            p = p + n;
        end
    end

end

