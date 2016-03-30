function [ trained_models ] = train_ann( models, datasets )
    %TRAIN_ANN Train one or more models on one or more datasets, results will be either one trained model or a cell array of trained models

    assert( numel(models) == 1 || numel(datasets) == 1, 'Training supports multiple models on one data set, or one data set for multiple models, but not both' );
    useParallelTraining = ( numel(models) > 1 || numel(datasets) > 1 );
    num_iterations = numel(models)*numel(datasets);
    trained_models = cell(num_iterations,1);
    total_epochs = count_total_epochs(models, datasets);
    
	tic;
    %p = ProgressBar( numel(models)*numel(datasets) );      %Per model progress bar
    if( useParallelTraining )
        p = ProgressBar( total_epochs );
        parfor n = 1:num_iterations
            % Use either the single dataset or pick the next one from the cell array of inputs.
            if(numel(datasets) == 1); DATA = datasets;    
            else                      DATA = cell2mat(datasets(n)); end;

            % Use either the single model or pick the next one from the cell array of inputs.
            if(numel(models) == 1);   MODEL = models;
            else                      MODEL = cell2mat(models(n)); end;

            % Set the model title for plotting to an IX number if it wasn't set.
            if( strcmp(MODEL.title, '') ); MODEL.title = sprintf('%d',n); end;
            
            % Turn off verbosity for parallel training, train, and update progress
            MODEL.verbosity = 'none';
            trained_models(n) = { train_network( MODEL, DATA, p ) };
            %p.progress;        % Moved into train_network, per epoch updates, this marked for deletion.
        end %END parfor
        p.stop;
    
    else
        % Train without parfor, faster for single training, slower for training multiple models.
        trained_models = { train_network( models, datasets ) };
        %p.progress;            % Moved into train_network, per epoch updates, this marked for deletion.
    end
    toc;

end

function [ total_epochs ] = count_total_epochs( models, datasets )
    total_epochs = 0;
    if( iscell(models) )
        for i = 1:numel(models)                                     % Count epochs in each model
            total_epochs = total_epochs + models{i}.num_epochs;
        end
    else
        total_epochs = models.num_epochs;
    end
    total_epochs = total_epochs * numel(datasets);              % Multiply by the # of datasets
end