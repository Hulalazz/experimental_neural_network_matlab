function [ predictions ] = predict_ann( trained_model, feature_data )
%PREDICT_ANN Summary of this function goes here
%   This function incomplete as of now. Note that it is not necessary if you define a cross validation set in your data, in this case predictions will be made and data available in the trained_model.Metrics struct that do everything necessary.

    [a, ~] = feedforward( trained_model, feature_data );
    output_layer = a{end};
    
    if( size(output_layer,2) > 1 )
        [~,label] = max(output_layer,[],2);      % class label format, not 1 of k, not sure, this might be useful, but other code is expecting 1 of k so I'm outputting it that way for now
        predictions = false(size(output_layer));
        max_indices = sub2ind(size(output_layer), (1:size(output_layer,1))', label);
        predictions(max_indices) = true;
    else
        predictions = round(output_layer);
    end
    
end

