function [ model ] = eg_sharing_past_average( model )                                       
    %EG_SHARING_PAST_AVERAGE Implements past average sharing for EG+/- update method, uses parameters model.eg_sharing, model.EG_sharing_alpha, past average sharing, which mixes weights from previous iterations.

    % TODO this is forcing a full copy of the weights/biases, bad performance issue
    
    weights_biases_flat = flatten_weights_biases( model, model.EG_sharing_inc_biases );
    
	% Update weights
    alpha = model.EG_sharing_alpha;
    weights_biases_flat_updated = ((1-alpha) .* weights_biases_flat) + (model.EG_sharing_past_weights_biases .* alpha);
    [ model.weights, model.biases ] = unflatten_weights_biases( model, weights_biases_flat_updated, model.EG_sharing_inc_biases );

    % Update past share - Average in the original weights in the past share data structure
    n = double(model.Scratch.batch_num);
    model.EG_sharing_past_weights_biases = (model.EG_sharing_past_weights_biases*n + weights_biases_flat) / (n+1);
    
end


