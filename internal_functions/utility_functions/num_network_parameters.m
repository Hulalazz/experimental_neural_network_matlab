function [ num_weights_biases ] = num_network_parameters( model )
    %NUM_NETWORK_PARAMETERS Computes the total number of network parameters (weights and biases) across all layers. Useful for flatting/unflattening the parameters.

    num_weights_biases = sum( [model.layer_sizes(1:end-1) .* model.layer_sizes(2:end) , model.layer_sizes(2:end)] ) * (strcmp(model.update_method,'EG+-')+1);
end

