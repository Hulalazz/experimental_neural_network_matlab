function [ a, z ] = feedforward( model, input, optional_predict_mode )
    % Input model (weights, biases, etc), and X's
    % Return the activations (post sigmoid), and z values (pre-sigmoid x*w calculations)
    
    %
    % Variable options parsing
    %
%    PREDICT_MODE = {'training', ...
%                    'predicting' };

%	p = inputParser;
%    addRequired( p, 'model',                            @(x)isstruct(x) );
%    addRequired( p, 'input',                            @(x)ismatrix(x) );
%    addOptional( p, 'predict_mode',     'predicting',   @(v)any(strcmp(PREDICT_MODE,v)) );    % Dropout weight adjustments are applied differently depending on whether we're in training mode (using a fraction of the neurons) or predicting mode (using all neurons and rescaling the weights).
%    parse( p, varargin{:} );

    % predict_mode options: 'training' | 'predicting'
    if( nargin >= 3 ); predict_mode = optional_predict_mode;
    else predict_mode = 'predicting'; end
    
    if(isa(model,'cell')); model = cell2mat(model); end
    
    a = cell(model.num_layers, 1);  % Hold activations
	z = cell(model.num_layers, 1);  % Hold weight inputs, z values at each later
    
    % Input layer
    a{1} = input;
    
    jl_embedding(1);            % Handle the JL embeddings if enabled
    dropout_training(1);        % Handle dropout additions at layer one (or do nothing if dropout isn't enabled)
    
	% Compute activations and z for each layer after input layer
    for l = 2:(model.num_layers)
        return_weights_biases_from_model(l-1);                               % Grab the weights and biases, there are some dependencies there so it's just cleaner to move this to its own subroutine to keep the code more readeable.
        z{l} = bsxfun( @plus, a{l-1} * weights, biases );
        
        if( use_softmax_at_this_layer(model, l) )       %TODO this is a work in progress and probably not as clean as it needs to ultimately be
            a{l} = softmax_function( z{l} );
        else
            %a{l} = sigmoid( z{l} );
            a{l} = model.transfer_function( z{l} );
        end
        
        jl_embedding(l);            % Handle the JL embeddings if enabled
        dropout_training(l);        % Handle dropout manipulations at lth layer (or do nothing if dropout isn't enabled)
    end

    
    %
    %
    % Nested Functions (parameters shared with the parent) - these functions implement various additional/optional 
    %                  features while keeping the core feedforward code clean and easy to follow
    %
    %
    
    % Perform JL projections if: (1) they're enabled, (2) they're enabled for this layer (e.g. not set to -1)
    function jl_embedding(jl_layer)
        if( ~isempty(model.jl_projection) && model.jl_projection(jl_layer) ~= -1 )
            %z{l} = z{l} * model.jl_projection_matricies{l} ./ sqrt(model.jl_k_layer_sizes(l));
            a{jl_layer} = (a{jl_layer} * model.jl_projection_matricies{jl_layer}) ./ sqrt(model.jl_projection(jl_layer)); %model.jl_scalefactor(jl_layer);
        end
    end
    
    
    % This function checks whether we are using EG+- or GD, if it's EG+- it combines the pos & neg weights and returns them, 
    % else just returns the normal weights used by GD.
    % This function is just here to clean up the case statements and organize them in one place for clarity.
    function return_weights_biases_from_model(wl)
        switch( model.update_method )
            case 'EG+-'
                weights = (model.weights.positive{wl} - model.weights.negative{wl});
            case 'GD'
                weights = model.weights{wl};
            otherwise
                assert( false, 'Update method not recognized.' );
        end
        biases = model.biases{wl};

        % When dropout is enabled and the prediction mode is predicting we need to rescale the weights.
        if( ~isempty(model.dropout_p) && strcmp(predict_mode,'predicting') ); weights = weights .* model.dropout_p(wl); end
    end % return_weights_biases_from_model
    

    function dropout_training(dl)
        if( ~isempty(model.dropout_p) && strcmp(predict_mode,'training') )
            a{dl} = model.Scratch.dropout_r{dl} .* a{dl};
        end
    end

end % feedforward




% This function determines if the given layer should use a softmax output or not.
% Current implementation only checks the last layer, but it's coded this way for easy extensibility
% and general code readability.
function [truefalse] = use_softmax_at_this_layer(model, layer)
    truefalse = model.use_softmax_output_layer && layer == model.num_layers;
end


