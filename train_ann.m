function [ trained_models ] = train_ann( models, datasets, varargin )
    %TRAIN_ANN Train one or more models on one or more datasets, results will be either one trained model or a cell array of trained models
    
	ip = inputParser;
    addRequired( ip, 'models' );
    addRequired( ip, 'datasets' );
    addOptional( ip, 'workers',      feature('numCores')+2,      @(x)isnumeric(x) );     % Max worker threads to use in parallel processing, default to # of cores + 2
    addOptional( ip, 'save',         'none',                     @(x)ischar(x) );        % Exports results to disk as they complete instead of as a return value.
    parse( ip, models, datasets, varargin{:} );
    
    useParallelTraining = ( numel(models) > 1 || numel(datasets) > 1 );     % Determines if we run in parallel or sequential mode, and in a few cases the optimal number of workers is tweaked for quick runs to process better.
    num_workers = ip.Results.workers;
    save_results = ~strcmp( ip.Results.save, 'none' );
    save_filepath = ip.Results.save;
    
    assert( numel(models) == 1 || numel(datasets) == 1, 'Training supports multiple models on one data set, or one data set for multiple models, but not both' );
    num_iterations = numel(models)*numel(datasets);
    trained_models = cell(num_iterations,1);
    total_steps = count_total_steps(models, datasets);

	tic;
    if( useParallelTraining )
        p = ProgressBar( total_steps );
        parfor ( n = 1:num_iterations, num_workers )
            % Use either the single dataset or pick the next one from the cell array of inputs.
            if(numel(datasets) == 1); DATA = datasets;    
            else                      DATA = cell2mat(datasets(n)); end;

            % Use either the single model or pick the next one from the cell array of inputs.
            if(numel(models) == 1);   MODEL = models;
            else                      MODEL = cell2mat(models(n)); end;

            % Set the model title for plotting to an IX number if it wasn't set.
            if( strcmp(MODEL.title, '') ); MODEL.title = sprintf('%d',n); end;
            
            % Turn off verbosity for parallel training, run training train, save results
            MODEL.verbosity = 'none';
            trained_model = train_network( MODEL, DATA, p );
            if( save_results );     parsave( trained_model, save_filepath, n );
            else                    trained_models(n) = { trained_model };      end
            
        end %END parfor
        p.stop;
    
    else
        % Train without parfor, faster for single training, slower for training multiple models.
        trained_models = { train_network( models, datasets ) };
        %p.progress;            % Moved into train_network, per epoch updates, this marked for deletion.
    end
    
    % Aggregate saved files
    if( save_results )
        trained_models={}; 
        aggregate_saved_files( save_filepath, num_iterations ); 
    end;
    toc;

end

function [ total_epochs ] = count_total_steps( models, datasets )
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

function aggregate_saved_files(save_filepath, num_iterations)
        output = matfile( sprintf('%s.mat', save_filepath) );
        output.trained_models = {};
        for n = 1:num_iterations
            load( sprintf('%s-%04d.mat', save_filepath, n) );        % Loads variable trained_model
            output.trained_models(1,n) = {trained_model};           % Save variable in aggregate file
            delete( sprintf('%s-%04d.mat', save_filepath, n) );      % Delete individual file
        end
end

function parsave( trained_model, pathfilename, n ) %#ok<INUSL>
    save( sprintf('%s-%04d.mat', pathfilename, n), 'trained_model' );
end