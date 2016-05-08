function [ weights, biases ] = unflatten_weights_biases( model, weights_and_biases_flat, optional_include_biases )
    %UNFLATTEN_WEIGHTS_BIASES Converts the output from flatten_weights_biases back into their standard hierarchical form.
    
% 	p = inputParser;
%     addRequired( p, 'model' );
%     addOptional( p, 'include_biases', true );
%     parse( p, model, varargin{:} );
    
    % Optional include biases
    if(nargin >= 3); include_biases = optional_include_biases;
    else include_biases = true; end;

    p = 1;
    switch model.update_method
        case 'EG+-'
            weights.positive = cell(model.num_layers-1,1);
            weights.negative = cell(model.num_layers-1,1);
        case 'GD'
            weights = cell(model.num_layers-1);
    end
	if( include_biases )
        biases = cell(model.num_layers-1,1);
    else
        biases = model.biases;      % Do nothing with the biases, just return what was already in the model.
    end

    
	for i = 1:model.num_layers-1
        % Weights
        n = model.layer_sizes(i);
        m = model.layer_sizes(i+1);
        switch model.update_method
            case 'EG+-'
                weights.positive{i} = reshape( weights_and_biases_flat(     (p):(p+n*m-1)   ), n, m );
                weights.negative{i} = reshape( weights_and_biases_flat( (p+n*m):(p+n*m*2-1) ), n, m );
                p = p + 2*n*m;
            case 'GD'
                weights{i} = reshape( weights_and_biases_flat( (p):(p+n*m-1) ), n, m ); %#ok<AGROW>
                p = p + n*m;
            otherwise; error( 'update_method not recognized' );
        end
        % Biases
        if( include_biases )
            m = model.layer_sizes(i+1);
            biases{i} = reshape( weights_and_biases_flat( (p):(p+m-1) ), 1, m );
            p = p + m;
        end
	end

end

