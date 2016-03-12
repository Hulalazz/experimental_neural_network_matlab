%% Models
model_quad_noreg     = initialize_model([784 30 10], 'mini_batch_size', 100, 'num_epochs', 30, 'cost_function', 'sum_of_squares', 'regularization', 'none', 'rng_seed', 777 )
model_quad_l2        = initialize_model([784 30 10], 'mini_batch_size', 100, 'num_epochs', 30, 'cost_function', 'sum_of_squares', 'regularization', 'L2', 'lambda', 1.3e-4, 'rng_seed', 777 )
model_centropy_noreg = initialize_model([784 30 10], 'mini_batch_size', 100, 'num_epochs', 30, 'cost_function', 'cross_entropy', 'regularization', 'none', 'rng_seed', 777 )
model_centropy_l2    = initialize_model([784 30 10], 'mini_batch_size', 100, 'num_epochs', 300, 'cost_function', 'cross_entropy', 'regularization', 'none', 'lambda', 1.3e-4, 'rng_seed', 777 )

model_centropy_n     = initialize_model([784 30 10], 'mini_batch_size', 100, 'num_epochs', 100, 'cost_function', 'cross_entropy', 'regularization', 'L2', 'lambda', 1.3e-4, 'rng_seed', 777, 'learning_rate', 0.3, 'weight_init_method', '1/sqrt(n)' )
model_EG = initialize_model([784 30 10], 'num_epochs', 100, 'update_method', 'EG+-', 'U', 10)
%% Create set of parameterized models
NUM_MODELS = numel(UVALUES);
models = {};
%master_model = initialize_model([784 30 10], 'mini_batch_size', 100, 'num_epochs', 60, 'cost_function', 'cross_entropy', 'regularization', 'L2', 'lambda', 0.013, 'rng_seed', 'shuffle', 'weight_init_method', 'gaussian-0-mean-1-std', 'learning_rate', 0.5 );
for i = 1:NUM_MODELS
    %m = master_model;
	m = initialize_model([784 30 10], 'num_epochs', 30, 'update_method', 'EG+-', 'U', UVALUES(i));

    %m.U = uvalues(i);
    models{end+1} = m;
end
clear NUM_MODELS i m master_model;


%% Train Multiple models using parfor
tic;
DATA = data_1k;
%models = {model_centropy_l2, model_quad_l2};

trained_models = cell(numel(models),1);
p = ProgressBar( numel(models) );
parfor i = 1:numel(models)
    m = cell2mat(models(i));
    m.verbosity = 'none';
    trained_models(i) = { trainann( m, DATA ) };
    p.progress;
end
p.stop;
clear to_train data i m p DATA
toc;

%% comparison training
train_centropy_noreg = trainSGD( model_centropy_noreg, data_40k );
train_centropy_l2    = trainSGD( model_centropy_l2, data_40k );

%%%%%% INCORPORATED INTO MAIN CODE BASE - DELETE THIS
% % %% Train using multiple datasets
% % tic;
% % DATA = data_1k_set;
% % trained_models = cell(numel(DATA),1);
% % p = ProgressBar(numel(DATA));
% % parfor i = 1:numel(DATA)
% % 
% %     m = model_centropy_noreg;
% %     m.verbosity = 'none';
% %     
% %     trained_models(i) = { trainSGD( m, DATA{i} ) };
% %     p.progress;
% % end
% % p.stop;
% % clear to_train data i m p
% % toc;

%% Data sets
train_features = training_data_nndl;
train_labels = training_labels_nndl;
cv_features = test_data_nndl;
cv_labels = test_labels_nndl;
data_1k_set = {};
for i = 1:50; r = randperm(42000); data_1k_set{end+1} = initialize_data( train_features(r(1:1000),:), train_labels(r(1:1000),:), 'cv_features', cv_features, 'cv_labels', cv_labels, 'rescale_method', 'none' ); end;
r = randperm(42000);
data_1k  = initialize_data(train_features(r(1:1000),:), train_labels(r(1:1000),:), 'cv_features', cv_features, 'cv_labels', cv_labels, 'rescale_method', 'none' )
data_5k  = initialize_data(train_features(r(1:5000),:), train_labels(r(1:5000),:), 'cv_features', cv_features, 'cv_labels', cv_labels, 'rescale_method', 'none' )
data_50k = initialize_data(train_features, train_labels, 'cv_features', cv_features, 'cv_labels', cv_labels, 'rescale_method', 'none' )
clear train_features train_labels cv_features cv_labels r i


%% Average multiple runs of a set of models into one 
% (eventually this should merge into a kfold functionality in initialize_data)
% This code isn't really a good approach to this problem, the iterations should be stored as an array under Metrics, and Metrics should be the aggregate values.
set_of_trained_models = trained_aggregates_EGvsGD_withnoise;
mean_of_models = cell( numel(set_of_trained_models), 1 );
for model_ix = 1:numel(set_of_trained_models{1})
    m = set_of_trained_models{1}{model_ix};
    mean_of_models{model_ix} = m;
    mean_of_models{model_ix}.Metrics.training_cost     = zeros( size(m.Metrics.training_cost) );
    mean_of_models{model_ix}.Metrics.training_accuracy = zeros( size(m.Metrics.training_accuracy) );
    mean_of_models{model_ix}.Metrics.cv_accuracy       = zeros( size(m.Metrics.cv_accuracy) );
    for fold_ix = 1:numel(set_of_trained_models)
        m = set_of_trained_models{fold_ix}{model_ix};
        mean_of_models{model_ix}.Metrics.training_cost     = mean_of_models{model_ix}.Metrics.training_cost +     ( m.Metrics.training_cost / numel(set_of_trained_models) );
        mean_of_models{model_ix}.Metrics.training_accuracy = mean_of_models{model_ix}.Metrics.training_accuracy + ( m.Metrics.training_accuracy / numel(set_of_trained_models) );
        mean_of_models{model_ix}.Metrics.cv_accuracy       = mean_of_models{model_ix}.Metrics.cv_accuracy +       ( m.Metrics.cv_accuracy / numel(set_of_trained_models) );
    end
end
