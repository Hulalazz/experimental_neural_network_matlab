function [ model ] = eg_sharing_past_average( model )
    %EG_SHARING_PAST_AVERAGE Implements past average sharing for EG+/- update method
    % Uses parameters model.eg_sharing, model.EG_sharing_alpha, model.EG_sharing_past_avg_agg_batches to implement
    % past average sharing, which mixes weights from previous iterations.

    weights_biases_flat = flatten_weights_biases( model );
    if( isempty(model.EG_sharing_past_weights_biases) )
        model.EG_sharing_past_weights_biases = zeros( num_network_parameters(model), ceil(double(model.Scratch.number_mini_batches)*double(model.num_epochs)/double(model.EG_sharing_past_avg_agg_batches)), 'like', model.biases{1} );
    end
    
	% Update weights with shared past (before storing the current weights)
    aggregate_columns = ceil((model.Scratch.batch_num-1) / model.EG_sharing_past_avg_agg_batches);
    fraction_prev_weights = model.EG_sharing_alpha * model.EG_sharing_past_avg_agg_batches / aggregate_columns;     % We make a simplifying assumption here that each column of past weights contain exactly model.EG_sharing_past_avg_agg_batches weights per column. This isn't true for the most recent column and we'll correct for that later.
    fraction_oversample_current_aggregate = model.EG_sharing_alpha * ((model.EG_sharing_past_avg_agg_batches-1) - mod(model.Scratch.batch_num-1, model.EG_sharing_past_avg_agg_batches)) / aggregate_columns; % Fraction to remove to correct the above simplifying assumption (we oversampled above by a small amount to simplify the math operation, this reverses that).
    
    if( aggregate_columns == 0 )
        past_share = weights_biases_flat .* model.EG_sharing_alpha;
    else
        past_share = sum( model.EG_sharing_past_weights_biases .* fraction_prev_weights, 2 ) ...
                      - ( model.EG_sharing_past_weights_biases(:,aggregate_columns) .* fraction_oversample_current_aggregate );
    end
    
    % Average in the original weights in the past share data structure
    model.EG_sharing_past_weights_biases(:,aggregate_columns+1) = (model.EG_sharing_past_weights_biases(:,aggregate_columns+1) + weights_biases_flat) ...
                                                                    ./ double(mod(model.Scratch.batch_num, model.EG_sharing_past_avg_agg_batches)+1);                 % If the column didn't have any previous entries just divid by 1, if there was a previous average, then devide by 2 to maintain the mean from iteration to iteration.
    
	% Update weights
    weights_biases_flat_updated = ((1-model.EG_sharing_alpha) .* weights_biases_flat) + past_share;
    [ model.weights, model.biases ] = unflatten_weights_biases( model, weights_biases_flat_updated );
end


