function [fig] = plot_n_runs_accuracy_cost( cell_array_of_traied_models, varargin )
    %PLOT_N_RUNS_ACCURACY_COST Plot all cv accuracy results from a set of runs, arguments: plot_training_accuracy, plot_cv_accuracy, plot_training_cost
    %   Input: A cell array of trained models
    
    %Suppress irrelevant compiler warnings 
    %#ok<*AGROW>

    p = inputParser;
    
    addRequired( p, 'models_to_plot',                               @(x)iscell(x) );
    addOptional( p, 'plot_training_accuracy',   true,               @(x)islogical(x) );
    addOptional( p, 'plot_cv_accuracy',         true,               @(x)islogical(x) );
    addOptional( p, 'plot_training_cost',       true,               @(x)islogical(x) );
    addOptional( p, 'hold_on',                  false,              @(x)isa(x,'matlab.ui.Figure') );        % give a graphics context (using gcf for example) to plot into
    
    parse( p, cell_array_of_traied_models, varargin{:} );    
    models_to_plot         = p.Results.models_to_plot;

    if( p.Results.hold_on == false ); fig = figure;
    else                              fig = figure( p.Results.hold_on ); end

    hold on;
    legend_text = {};
    for mod = 1:numel(models_to_plot)
        m = models_to_plot{mod};
        if( p.Results.plot_training_accuracy )
            line = plot( m.Metrics.training_accuracy, 'LineWidth', 2, 'LineStyle', '--' ); 
            legend_text{end+1} = sprintf('Train Accuracy [%s]',m.title);
            label(line, m.title, 'location', 'right');
        end
        if( p.Results.plot_cv_accuracy )
            line = plot( m.Metrics.cv_accuracy, 'LineWidth', 1 );
            legend_text{end+1} = sprintf('CV Accuracy [%s]',m.title);
            label(line, m.title, 'location', 'center');
        end
        if( p.Results.plot_training_cost )
            line = plot( mat2gray(aggregate_data(m.Metrics.training_cost, numel(m.Metrics.training_accuracy))) );
            legend_text{end+1} = sprintf('Training cost [%s]',m.title);
            label(line, m.title, 'location', 'right');
        end
    end
    h_legend = legend(legend_text, 'Location','southwest');
    set(h_legend,'FontSize',14);
    xlabel('Epochs of training set');
    ylabel('Accuracy');
    hold off;


end

