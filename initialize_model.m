function [ model ] = initialize_model( layer_sizes, varargin )
    COST_FUNCTION_OPTIONS =  {'sum_of_squares', ...
                              'cross_entropy', ...
                              'loglikelihood' };
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
	EG_SHARING_OPTIONS =     {'none', ...
                              'past_average', ...
                              'decaying_past', ...
                              'fixed_regularization' };
    TRANSFER_FUNCTION_NAME = {'sigmoid', ...
                              'tanh', ...
                              'relu', ...
                              'leakyrelu' };
    LEARNING_RATES         = containers.Map(COST_FUNCTION_OPTIONS,  [ 3.0, 0.5, 1.0 ] );  % Default learning rates per cost function
    COST_FUNCTIONS_COST    = containers.Map(COST_FUNCTION_OPTIONS,  { @cost_quadratic,       @cost_cross_entropy,       @cost_log_likelihood       } );
    COST_FUNCTIONS_DELTA   = containers.Map(COST_FUNCTION_OPTIONS,  { @cost_quadratic_delta, @cost_cross_entropy_delta, @cost_log_likelihood_delta } );
    TRANSFER_FUNCTIONS     = containers.Map(TRANSFER_FUNCTION_NAME, { @sigmoid,       @tanh,  @relu,      @leakyrelu       } );
    TRANSFER_FUNCTIONS_INV = containers.Map(TRANSFER_FUNCTION_NAME, { @sigmoid_prime, @atanh, @relu_prime @leakyrelu_prime } );
    
    %**************************************************************
    % Input parameters, basic validation, and parameter parsing
    % Default values
    %**************************************************************
    p = inputParser;
    
    addRequired( p, 'layer_sizes',      @(x)ismatrix(x) );

    addOptional( p, 'plot_title',                       ''                  );                                                  % Title text used in plots, uses index when unspecified.
    addOptional( p, 'update_method',                    'GD',               @(v)any(strcmp(UPDATE_METHOD,v)) );
    addOptional( p, 'U',                                40,                 @(x)strcmp(x,'unnormalized')||isnumeric(x) );       % Used by EG+- update method
	addOptional( p, 'EG_sharing',                       'none',             @(v)any(strcmp(EG_SHARING_OPTIONS,v)) );            % Past weight sharing for EG updates
    addOptional( p, 'EG_sharing_alpha',                 0.1,                @(x)isnumeric(x) );                                 % Used by EG+- when sharing is enabled
%    addOptional( p, 'EG_sharing_decay_q',               0.2,                @(x)isnumeric(x) );                                 % Used by EG+- when sharing with decaying past is enabled
    addOptional( p, 'EG_sharing_inc_biases',            false,              @(x)islogical(x) );                                 % Used by EG+- to share weights and biases both (true) or just weights (false)
    addOptional( p, 'transfer_function',                'sigmoid',          @(v)any(strcmp(TRANSFER_FUNCTION_NAME,v)) );
    addOptional( p, 'learning_rate',                    0,                  @(x)isnumeric(x) );
    addOptional( p, 'num_epochs',                       30,                 @(x)isnumeric(x) );
    addOptional( p, 'mini_batch_size',                  100,                @(x)isnumeric(x) );
    addOptional( p, 'cost_function',                    'cross_entropy',    @(v)any(strcmp(COST_FUNCTION_OPTIONS,v)) );
    addOptional( p, 'initial_weights',                  [],                 @(x)ismatrix(x) );
    addOptional( p, 'initial_biases',                   [],                 @(x)ismatrix(x) );
    addOptional( p, 'weight_init_method',               '1/sqrt(n)',        @(v)any(strcmp(WEIGHT_INIT_METHODS,v)) );
    addOptional( p, 'weight_bias_data_type',            single([]),         @(x)isempty(x) && isfloat(x) );
    addOptional( p, 'regularization',                   'L2',               @(v)any(strcmp(REGULARIZATION_OPTIONS,v)) );
    addOptional( p, 'lambda',                           1.0e-4,             @(x)isnumeric(x) );
    addOptional( p, 'monitor_training_cost',            true,               @(x)islogical(x) );
    addOptional( p, 'monitor_accuracy',                 true,               @(x)islogical(x) );
    addOptional( p, 'monitor_delta_norm',               true,               @(x)islogical(x) );
    addOptional( p, 'store_weight_history',             false,              @(x)islogical(x) );                                 % Stores each mini-batch weight in Metrics, warning, can be VERY large dataset
    addOptional( p, 'verbosity',                        'epoch',            @(v)any(strcmp(VERBOSITY,v)) );
    addOptional( p, 'rng_seed',                         'shuffle',          @(x)strcmp(x,'shuffle')||isnumeric(x) );            % May be 'shuffle' for random, or a fixed integer >0
    addOptional( p, 'noise_function',                   'none',             @(f)isa(f,'function_handle') || strcmp(f,'none') );
    addOptional( p, 'use_softmax_output_layer',         false,              @(x)islogical(x) );                                 % Automatically enabled if loglikelihood cost function is used.
    addOptional( p, 'debug_check_numerical_gradients',  false,              @(x)islogical(x) );                                 % Enables debug checking of numerical gradients on each backprop step, very slow
    addOptional( p, 'dropout_p',                        [],                 @(x)isnumeric(x)||ismatrix(x) );                    % p value (single) or array of per-layer p-values for dropout. Setting this enables dropout. [] for no dropout
    addOptional( p, 'jl_projection',                    [],                 @(x)isvector(x) );                                  % Defines JL projections per layer and enables JL projection processing
    
    parse( p, layer_sizes, varargin{:} );

    
    %************************************************
    % Build model struct with appropriate settings
    %************************************************
    % Required parameters
    model.layer_sizes                       = p.Results.layer_sizes;
    model.num_layers                        = numel(p.Results.layer_sizes);
    
    % Optional parameters
    model.learning_rate                     = p.Results.learning_rate;
    model.num_epochs                        = p.Results.num_epochs;
    model.mini_batch_size                   = p.Results.mini_batch_size;
    model.cost_function                     = p.Results.cost_function;
    model.cost_function_cost                = COST_FUNCTIONS_COST( p.Results.cost_function );
    model.cost_function_delta               = COST_FUNCTIONS_DELTA( p.Results.cost_function );
    model.transfer_function                 = TRANSFER_FUNCTIONS( p.Results.transfer_function );
    model.transfer_function_inv             = TRANSFER_FUNCTIONS_INV( p.Results.transfer_function );
    model.regularization                    = p.Results.regularization;
    model.lambda                            = p.Results.lambda;
    model.update_method                     = p.Results.update_method;
    model.use_softmax_output_layer          = p.Results.use_softmax_output_layer;
    model.debug_check_numerical_gradients   = p.Results.debug_check_numerical_gradients;
    
    model.monitor_training_cost             = p.Results.monitor_training_cost;
    model.monitor_accuracy                  = p.Results.monitor_accuracy;
    model.monitor_delta_norm                = p.Results.monitor_delta_norm;
    model.store_weight_history              = p.Results.store_weight_history;
    model.verbosity                         = p.Results.verbosity;
    model.rng_seed                          = p.Results.rng_seed;
    model.title                             = p.Results.plot_title;
    model.dropout_p                         = p.Results.dropout_p;
    model.noise_function                    = p.Results.noise_function;
%    model.callbacks_post_update             = {};                                               % Callback functions post mini batch update
    model.Scratch.batch_num                 = 1;                                                % The batch iteration number, incremented after each minibatch update
    model.jl_projection                     = p.Results.jl_projection;
    
    % Use random number seed for initializations, rng takes 'shuffle' or the value passed to rng_seed
    rng(p.Results.rng_seed);
    
    % Initialize the Metrics struct where metric data is stored if the model calls for it.
    model.Metrics = struct;
    if( p.Results.monitor_training_cost ); model.Metrics.training_cost = {}; end
    if( p.Results.monitor_accuracy )
        model.Metrics.training_accuracy = {};
        model.Metrics.cv_accuracy = {};
    end
    if( p.Results.store_weight_history ); model.Metrics.weight_bias_history = {}; end;
    if( p.Results.monitor_delta_norm ); model.Metrics.delta_norm = {}; end;
    
    % Initialize weights and biases
    if( isempty(p.Results.initial_weights) )
        model = initialize_weights_and_biases( model, p );
    else
        assert( ~isempty(p.Results.initial_biases), 'If initial weights are provided with the ''initial_weights'' options, then initial biases must also be provided with ''initial_biases''' );
        assert( strcmp(class(p.Results.initial_weights), class(p.Results.initial_biases)), 'Pre-initialized weights and biases classes don''t match.' );
        model.weights = p.Results.initial_weights;
        model.biases = p.Results.initial_biases;
    end
    
        % Parameters specific to EG+/- update method (depends on initialization of weights/biases)
    if( strcmp(p.Results.update_method,'EG+-') )
        assert( strcmp(p.Results.U,'unnormalized') || p.Results.U > 0, 'U parameter must be present when using EG+- updates, it should be a positive real number' );
        model.U = p.Results.U;
        model.EG_sharing                        = p.Results.EG_sharing;
        model.EG_sharing_alpha                  = p.Results.EG_sharing_alpha;
        model.EG_sharing_inc_biases             = p.Results.EG_sharing_inc_biases;
        model.EG_sharing_past_weights_biases    = flatten_weights_biases( model, p.Results.EG_sharing_inc_biases );

        
        % EG+- no regularization is used, it's intrisically built into the update method via the U parameter
        model.regularization = 'none';
        if( all(~strcmp('regularization', p.UsingDefaults)) ); warning('A regularization method was specified, however update method EG+- does not use this parameter. Regularization is intrinsic via the U parameter for EG+-. The regularization parameter has been nullified.'); end;
        
    end
    

    if( ~strcmp(p.Results.regularization,'none') && p.Results.lambda==0 ); warning('Regularization has been configured, but labmda is set to 0, effectively nullifying regularization'); end;
    if( strcmp(p.Results.regularization, 'none') ); model.lambda = 0; end;
    
    % Initialize learning rate based on the cost function if it's not defined
    if( p.Results.learning_rate == 0 ); model.learning_rate = LEARNING_RATES(p.Results.cost_function); end;
    
    % Log likelihood cost function - when this cost function is used enable softmax output layer
    if( strcmp(p.Results.cost_function, 'loglikelihood') )
        model.use_softmax_output_layer = true;
    end
    
    % Dropout, validation and set the model parameter
    if( ~isempty(p.Results.dropout_p) )
        if( isnumeric(p.Results.dropout_p) )
            model.dropout_p = p.Results.dropout_p .* ones( 1, model.num_layers );
            model.dropout_p(1) = 1;
        elseif( ismatrix(p.Results.dropout_p) )
            assert( size(p.Results.dropout_p, 2) == model.num_layers );
            model.dropout_p = p.Results.dropout_p;
        end
    end
    
    % JL projections
    if( ~isempty(p.Results.jl_projection) )
        if( ~isa(p.Results.weight_bias_data_type, 'double') ); warning('JL projections may be unstable if used with single precision weights.'); end;
        model.jl_projection_matricies = {};
        for i = 1:numel(model.jl_projection)
            if( model.jl_projection(i) ~= -1 )
                model.jl_projection_matricies{i} = binornd(1, 0.5, [model.layer_sizes(i), model.jl_projection(i)])*2-1;
                model.jl_projection_inverses{i}  = pinv( model.jl_projection_matricies{i} );
            end
        end
    end
    
    
end


function [ model ] = initialize_weights_and_biases( model, p )
    % Layer sizes from/to for weight matrices, may vary depending on options such as JL transformations
	layer_sizes_to = p.Results.layer_sizes;
    if( ~isempty(p.Results.jl_projection) )
        use_jl = p.Results.jl_projection ~= -1;
        layer_sizes_from = p.Results.layer_sizes.*(~use_jl) + p.Results.jl_projection.*(use_jl);
    else
        layer_sizes_from = p.Results.layer_sizes;
    end
    
    n_layers = numel(p.Results.layer_sizes);
    model.weight_init_method = p.Results.weight_init_method;

    if( strcmp(p.Results.update_method, 'EG+-') )
        model.weights.positive = cell( n_layers-1, 1 );
        model.weights.negative = cell( n_layers-1, 1 );
        model.biases = cell( n_layers-1, 1 );
        for l = 1:(n_layers-1)
            model.weights.positive{l} = 1/sqrt(layer_sizes_from(l)) .* rand( layer_sizes_from(l), layer_sizes_to(l+1), 'like', p.Results.weight_bias_data_type );
            model.weights.negative{l} = 1/sqrt(layer_sizes_from(l)) .* rand( layer_sizes_from(l), layer_sizes_to(l+1), 'like', p.Results.weight_bias_data_type );
            model.biases{l}   = rand( 1, layer_sizes_to(l+1), 'like', p.Results.weight_bias_data_type );

            % Normalize to U
            if( ~strcmp(p.Results.U,'unnormalized') )
                sum_inputs_to_neuron = sum(model.weights.positive{l},1) + sum(model.weights.negative{l},1);
                model.weights.positive{l} = p.Results.U .* bsxfun(@rdivide, model.weights.positive{l}, sum_inputs_to_neuron);
                model.weights.negative{l} = p.Results.U .* bsxfun(@rdivide, model.weights.negative{l}, sum_inputs_to_neuron);
            end
        end
        
    elseif( strcmp(p.Results.weight_init_method, '1/sqrt(n)') )
        % 1/sqrt(n) weights normalization method - Normally distributed weights with 0 mean and standard deviation of 1/sqrt(n)
        for l = 1:(n_layers-1)
            model.weights{l} = 1/sqrt(layer_sizes_from(l)) .* randn( layer_sizes_from(l), layer_sizes_to(l+1), 'like', p.Results.weight_bias_data_type );
            model.biases{l}  = randn( 1,                                                  layer_sizes_to(l+1), 'like', p.Results.weight_bias_data_type  );
        end
    elseif( strcmp(p.Results.weight_init_method, 'gaussian-0-mean-1-std') )
        % Gaussian initialization with 0 mean, 1 standard deviation
        for l = 1:(n_layers-1)
            model.weights{l} = randn( layer_sizes_from(l), layer_sizes_to(l+1), 'like', p.Results.weight_bias_data_type );
            model.biases{l}  = randn( 1,                   layer_sizes_to(l+1), 'like', p.Results.weight_bias_data_type  );
        end
    else
        assert( false, 'Weight initialization method not recognized' );
    end
end

