function [ num_params ] = num_network_parameters( model, optional_include_biases  )
    %NUM_NETWORK_PARAMETERS Computes the total number of network parameters (weights and biases) across all layers. Useful for flatting/unflattening the parameters.

% 	p = inputParser;
%     addRequired( p, 'model' );
%     addOptional( p, 'include_biases', true );           % Function computes either just weights or weights + biases
%     parse( p, model, varargin{:} );

    if(nargin > 1); include_biases = optional_include_biases;
    else include_biases = true; end;

    % Count the weights
    num_params = sum( model.layer_sizes(1:end-1) .* model.layer_sizes(2:end) );
    
    % Conditionally count the biases
    if( include_biases ) 
        num_params = num_params + sum( model.layer_sizes(2:end) );
    end
    
    % Double EG+- params
    if( strcmp(model.update_method,'EG+-') )
        num_params = num_params * 2;
    end
end

