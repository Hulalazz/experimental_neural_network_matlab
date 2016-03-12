function [ model ] = initialize_model( layer_sizes, varargin )
    COST_FUNCTION_OPTIONS =  {'sum_of_squares', ...
                              'cross_entropy' };
    WEIGHT_INIT_METHODS =    {'1/sqrt(n)', ...
                              'gaussian-0-mean-1-std'};
    VERBOSITY =              {'none', ...           % This should really be handled with an isVerboseEnough function.
                              'epoch', ...
                              'debug' };
	UPDATE_METHOD =          {'GD', ...
                              'EG+-' };
    REGULARIZATION_OPTIONS = {'none', ...
                              'L1', ...
                              'L2' };
    LEARNING_RATES      = containers.Map(COST_FUNCTION_OPTIONS, [ 3.0  0.5 ] );  % Default learning rates per cost function
    COST_FUNCTIONS_COST  = containers.Map(COST_FUNCTION_OPTIONS, { @cost_quadratic,       @cost_cross_entropy       } );
    COST_FUNCTIONS_DELTA = containers.Map(COST_FUNCTION_OPTIONS, { @cost_quadratic_delta, @cost_cross_entropy_delta } );
    
    %**************************************************************
    % Input parameters, basic validation, and parameter parsing
    % Default values
    %**************************************************************
    p = inputParser;
    
    addRequired( p, 'layer_sizes',      @(x)ismatrix(x) );

    addOptional( p, 'plot_title',               ''                  );                                                  % Title text used in plots, uses index when unspecified.
    addOptional( p, 'update_method',            'GD',               @(v)any(strcmp(UPDATE_METHOD,v)) );
    addOptional( p, 'U',                        40,                 @(x)isnumeric(x) );                                 % Used by EG+- update method
    addOptional( p, 'learning_rate',            0,                  @(x)isnumeric(x) );
    addOptional( p, 'num_epochs',               30,                 @(x)isnumeric(x) );
    addOptional( p, 'mini_batch_size',          100,                @(x)isnumeric(x) );
    addOptional( p, 'cost_function',            'sum_of_squares',   @(v)any(strcmp(COST_FUNCTION_OPTIONS,v)) );
    addOptional( p, 'initial_weights',          [],                 @(x)ismatrix(x) );
    addOptional( p, 'initial_biases',           [],                 @(x)ismatrix(x) );
    addOptional( p, 'weight_init_method',       '1/sqrt(n)',        @(v)any(strcmp(WEIGHT_INIT_METHODS,v)) );
    addOptional( p, 'weight_bias_data_type',    single([]),         @(x)isempty(x) && isfloat(x) );
    addOptional( p, 'regularization',           'L2',               @(v)any(strcmp(REGULARIZATION_OPTIONS,v)) );
    addOptional( p, 'lambda',                   1.0e-4,             @(x)isnumeric(x) );
    addOptional( p, 'monitor_training_cost',    true,               @(x)islogical(x) );
    addOptional( p, 'monitor_accuracy',         true,               @(x)islogical(x) );
    addOptional( p, 'verbosity',                'epoch',            @(v)any(strcmp(VERBOSITY,v)) );
    addOptional( p, 'rng_seed',                 'shuffle',          @(x)strcmp(x,'shuffle')||isnumeric(x) );            % May be 'shuffle' for random, or a fixed integer >0
    addOptional( p, 'noise_function',           'none',             @(f)isa(f,'function_handle') || strcmp(f,'none') );
    
    parse( p, layer_sizes, varargin{:} );

    
    %************************************************
    % Build model struct with appropriate settings
    %************************************************
    % Required parameters
    model.layer_sizes = p.Results.layer_sizes;
    model.num_layers = numel(p.Results.layer_sizes);
    
    % Optional parameters
    model.learning_rate         = p.Results.learning_rate;
    model.num_epochs            = p.Results.num_epochs;
    model.mini_batch_size       = p.Results.mini_batch_size;
    model.cost_function         = p.Results.cost_function;
    model.regularization        = p.Results.regularization;
    model.lambda                = p.Results.lambda;
    model.update_method         = p.Results.update_method;
    
    model.monitor_training_cost = p.Results.monitor_training_cost;
    model.monitor_accuracy      = p.Results.monitor_accuracy;
    model.verbosity             = p.Results.verbosity;
    model.rng_seed              = p.Results.rng_seed;
    model.title                 = p.Results.plot_title;
    model.noise_function        = p.Results.noise_function;
    
    % Use random number seed for initializations, rng takes 'shuffle' or the value passed to rng_seed
    rng(p.Results.rng_seed);
    
    % Parameters specific to EG+/- update method
    if( strcmp(p.Results.update_method,'EG+-') )
        assert( p.Results.U ~= 0, 'U parameter must be present when using EG+- updates, it should be a positive real number' );
        model.U = p.Results.U;
    end
    
    % Initialize the Metrics struct where metric data is stored if the model calls for it.
    if( p.Results.monitor_training_cost || p.Results.monitor_accuracy ); model.Metrics = struct; end
    if( p.Results.monitor_training_cost ); model.Metrics.training_cost = {}; end
    if( p.Results.monitor_accuracy )
        model.Metrics.training_accuracy = {};
        model.Metrics.cv_accuracy = {};
    end
    
    % Initialize weights and biases
    if( isempty(p.Results.initial_weights) )
        model = initialize_weights_and_biases( model, p );
    else
        assert( ~isempty(p.Results.initial_biases), 'If initial weights are provided with the ''initial_weights'' options, then initial biases must also be provided with ''initial_biases''' );
        assert( strcmp(class(p.Results.initial_weights), class(p.Results.initial_biases)), 'Pre-initialized weights and biases classes don''t match.' );
        model.weights = p.Results.initial_weights;
        model.biases = p.Results.initial_weights;
    end
    
    if( ~strcmp(p.Results.regularization,'none') && p.Results.lambda==0 ); warning('Regularization has been configured, but labmda is set to 0, effectively nullifying regularization'); end;
    if( strcmp(p.Results.regularization, 'none') ); model.lambda = 0; end;
    
    % Initialize learning rate based on the cost function if it's not defined
    if( p.Results.learning_rate == 0 ); model.learning_rate = LEARNING_RATES(p.Results.cost_function); end;
    % Set up cost functions
    model.cost_function_cost  = COST_FUNCTIONS_COST(p.Results.cost_function);
    model.cost_function_delta = COST_FUNCTIONS_DELTA(p.Results.cost_function);
    
    % Sanity check intput for EG+-, no regularization is used, it's intrisically built into the update method via the U parameter
	if( strcmp(p.Results.update_method,'EG+-') )
        model.regularization = 'none';
        if( all(~strcmp('regularization', p.UsingDefaults)) ); warning('A regularization method was specified, however update method EG+- does not use this parameter. Regularization is intrinsic via the U parameter for EG+-. The regularization parameter has been nullified.'); end;
	end

end

function [ model ] = initialize_weights_and_biases( model, p )
	layer_sizes = p.Results.layer_sizes;
    n_layers = numel(layer_sizes);
    model.weight_init_method = p.Results.weight_init_method;

    if( strcmp(p.Results.update_method, 'EG+-') )
        model.weights.positive = cell( n_layers-1, 1 );
        model.weights.negative = cell( n_layers-1, 1 );
        model.biases = cell( n_layers-1, 1 );
        for l = 1:(numel(layer_sizes)-1)
            model.weights.positive{l} = 1/sqrt(layer_sizes(l)) .* rand( layer_sizes(l), layer_sizes(l+1), 'like', p.Results.weight_bias_data_type );
            model.weights.negative{l} = 1/sqrt(layer_sizes(l)) .* rand( layer_sizes(l), layer_sizes(l+1), 'like', p.Results.weight_bias_data_type );
            model.biases{l}   = rand( 1, layer_sizes(l+1), 'like', p.Results.weight_bias_data_type );

            % Normalize to U
            sum_inputs_to_neuron = sum(model.weights.positive{l},1) + sum(model.weights.negative{l},1);
            model.weights.positive{l} = model.U .* bsxfun(@rdivide, model.weights.positive{l}, sum_inputs_to_neuron);
            model.weights.negative{l} = model.U .* bsxfun(@rdivide, model.weights.negative{l}, sum_inputs_to_neuron);
        end
        
    elseif( strcmp(p.Results.weight_init_method, '1/sqrt(n)') )
        % 1/sqrt(n) weights normalization method - Normally distributed weights with 0 mean and standard deviation of 1/sqrt(n)
        for l = 1:(numel(layer_sizes)-1)
            model.weights{l} = 1/sqrt(layer_sizes(l)) .* randn( layer_sizes(l), layer_sizes(l+1), 'like', p.Results.weight_bias_data_type );
            model.biases{l}  = randn( 1,                                        layer_sizes(l+1), 'like', p.Results.weight_bias_data_type  );
        end
    elseif( strcmp(p.Results.weight_init_method, 'gaussian-0-mean-1-std') )
        % Gaussian initialization with 0 mean, 1 standard deviation
        for l = 1:(numel(layer_sizes)-1)
            model.weights{l} = randn( layer_sizes(l), layer_sizes(l+1), 'like', p.Results.weight_bias_data_type );
            model.biases{l}  = randn( 1,              layer_sizes(l+1), 'like', p.Results.weight_bias_data_type  );
        end
    else
        assert( false, 'Weight initialization method not recognized' );
    end
end

