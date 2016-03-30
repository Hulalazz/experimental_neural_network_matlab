
%% Plot accuracy/CV of two runs
trained_sets = trained_models;
LEGEND       = {'Training', 'Cross validation'};
for i = 1:numel(trained_sets)
    figure; hold on; 
    plot( cell2mat(trained_sets{i}.Metrics.training_accuracy) ); 
    plot( cell2mat(trained_sets{i}.Metrics.cv_accuracy) ); 
    legend(LEGEND{1}, LEGEND{2}); 
    title(trained_sets{i}.regularization, 'Interpreter','none');
    ylim([0.6 1])
    hold off;
end
clear trained_sets LEGEND i


%% Get last accuracy Metrics from a set of trained models & plot them
% try getField to condense this stuff to 1 line
tmp = cell2mat(trained_models);
tmp = cell2mat( {tmp.Metrics} );
tmp = {tmp.cv_accuracy};
cv_accuracy = zeros(numel(tmp),1);

for i = 1:numel(tmp)
    cv_accuracy(i) = mean(cell2mat( tmp{i}(end-5:end) )); 
end
figure; plot( UVALUES, cv_accuracy ); xlabel('U Values'); ylabel('cv accuracy');
clear tmp cv_accuracy

