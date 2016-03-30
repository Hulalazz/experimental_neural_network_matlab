function [ model ] = train_network( model, data, progress_bar )

    num_samples         = size(data.feature_data, 1);
    assert( num_samples < intmax('uint32') );
    batch_size          = model.mini_batch_size;
    number_mini_batches = floor( num_samples / batch_size );    % Take the floor and combine the final batch into a larger than normal batch to account for leftover samples that don't make a full batch
    learning_rate       = model.learning_rate;
    num_epochs          = model.num_epochs;
    rng_seed            = model.rng_seed;
    
    rng(rng_seed);      % Seed the random number generator up front, note this was set in the model parameters as 'shuffle' or an integer value
    
    assert( number_mini_batches > 0, ...                        % Sanity check
            sprintf('Number of mini batches is zero with a batch size of %d and number of samples %d\n', batch_size, num_samples ) );
    assert( size(data.feature_data,1) == size(data.labels,1), 'Training data length does not match label data length' );
    if( ~isempty(data.cv_feature_data) ); assert( size(data.cv_feature_data,1) == size(data.cv_labels,1), 'Optional CV test data size does not match optional CV training data size' ); end;

    for j = 1:num_epochs
        samples_shuffle = cast( randperm(num_samples), 'uint32' );
    
        for i = 1:number_mini_batches
            fromix = (i-1)*batch_size + 1;
            if( i == number_mini_batches ); toix = num_samples;
            else                            toix = i * batch_size;
            end;
            
            feature_data = data.feature_data(samples_shuffle(fromix:toix), :);
            label_data   = data.labels(samples_shuffle(fromix:toix), :);
            
            % Apply noise function if one was specified in the model options
            if( isa(model.noise_function, 'function_handle') )
                feature_data = model.noise_function(feature_data);
            end
            
            % Run one batch update
            model = update_mini_batch( model, feature_data, label_data, learning_rate  );
        end
        
        % Metrics
        if( model.monitor_accuracy && ~isempty(data.cv_feature_data) )
            %feedforward_features = (model,data.feature_data);
            %[~, predict_X_train] = max(feedforward_features{end}, [], 2);
            %[~, actual_y_train]  = max(data.labels, [], 2);
            %feedforward_labels   = feedforward(model,data.cv_feature_data);
            %[~, predict_X_cv]    = max(feedforward_labels{end}, [], 2);
            %[~, actual_y_cv]     = max(data.cv_labels, [], 2);

            %training_accuracy    = sum(predict_X_train==actual_y_train)/size(predict_X_train,1);
            %cv_accuracy          = sum(predict_X_cv==actual_y_cv)/size(predict_X_cv,1);
            
            training_accuracy     = sum( all(predict_ann(model,data.feature_data) == data.labels,2) ) / size(data.labels,1);
            cv_accuracy           = sum( all(predict_ann(model,data.cv_feature_data) == data.cv_labels,2) ) / size(data.cv_labels,1);

            if( strcmp(model.verbosity, 'epoch') || strcmp(model.verbosity, 'debug') )
                fprintf( 'Epoch %03i, Training Accuracy: %.3f, CV Accuracy: %.3f\n', j, training_accuracy, cv_accuracy );
            end
            model.Metrics.training_accuracy{end+1}  = training_accuracy;
            model.Metrics.cv_accuracy{end+1}        = cv_accuracy;
        else
            if( strcmp(model.verbosity, 'epoch') || strcmp(model.verbosity, 'debug') )
                fprintf( 'Epoch %j complete\n', j );
            end
        end % end Metrics section
        
        if( nargin >= 3 ) progress_bar.progress; end;       % Progress bar update per epoch

    end % end epoch loop
    
    % Convert metrics from cell arrays to standard matricies.
    model.Metrics.training_cost     = cell2mat( model.Metrics.training_cost );
    model.Metrics.training_accuracy = cell2mat( model.Metrics.training_accuracy );
    model.Metrics.cv_accuracy       = cell2mat( model.Metrics.cv_accuracy );
end

