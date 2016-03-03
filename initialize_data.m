function [ data ] = initialize_data( feature_data, labels, varargin )
    %INITIALIZE_DATA Options: rescaling_method = {'none','standardize','rescale'}
    
    FEATURE_SCALING_METHODS = {'none', ...
                               'standardize', ...
                               'rescale' };

    p = inputParser;
    
    addRequired( p, 'feature_data',     @(x)ismatrix(x) );
    addRequired( p, 'labels',           @(x)ismatrix(x) );

    addOptional( p, 'rescale_method',	'none',     @(v)any(strcmp(FEATURE_SCALING_METHODS,v)) );
    addOptional( p, 'cv_features',      [],         @(x)ismatrix(x) );
    addOptional( p, 'cv_labels',        [],         @(x)ismatrix(x) );
    
    parse( p, feature_data, labels, varargin{:} );

	% Input validation of CV optional features
	assert( (isempty(p.Results.cv_features) && isempty(p.Results.cv_labels)) || (~isempty(p.Results.cv_features) && ~isempty(p.Results.cv_labels)), 'If cv_features and cv_labels must *both* be included, or *neither*. Only one of the two was passed.' );

    % Rescaling
    switch p.Results.rescale_method
        case 'standardize'
            data.feature_data    = zscore(p.Results.feature_data);
            data.cv_feature_data = zscore(p.Results.cv_features);
        case 'rescale'
            data.feature_data    = mat2gray(p.Results.feature_data);
            data.cv_feature_data = mat2gray(p.Results.cv_features);
        case 'none'
            % Warn if rescaling not applied and is likely needed
            if( max(std(p.Results.feature_data)) > 3 )
                warning('Feature data does not appear scaled, this is only a warning and will not affect operations, but typically features scaling is necessary. Use ''feature_scaling_method'' to perform feature scaling in this init method.' );
            end
            
            data.feature_data    = p.Results.feature_data;
            data.cv_feature_data = p.Results.cv_features;

    end
    
    data.labels = p.Results.labels;
	data.cv_labels = p.Results.cv_labels;

    data.rescale_method = p.Results.rescale_method;


end

