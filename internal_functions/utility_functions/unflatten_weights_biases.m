function [ weights, biases ] = unflatten_weights_biases( model, weights_and_biases_flat )
    %UNFLATTEN_WEIGHTS_BIASES Converts the output from flatten_weights_biases back into their standard hierarchical form.
    p = 1;
    switch model.update_method
        case 'EG+-'
            weights.positive = cell(model.num_layers-1,1);
            weights.negative = cell(model.num_layers-1,1);
        case 'GD'
            weights = cell(model.num_layers-1);
    end
	biases = cell(model.num_layers-1,1);

    
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
        m = model.layer_sizes(i+1);
        biases{i} = reshape( weights_and_biases_flat( (p):(p+m-1) ), 1, m );
        p = p + m;
	end

end

