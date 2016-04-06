function [ num_grad_w, num_grad_b ] = numericalGradients( model, X, y )
    % Compute numerical gradients for debugging purposes

    num_grad_w = cell(model.num_layers-1,1);
    num_grad_b = cell(model.num_layers-1,1);
    h = 1e-4;
    
    a = feedforward(model,X);
    C = @(model,X,y) (1/2)*sum(sum((y-a{end}).^2));
    
    for l = 1:model.num_layers-1
        num_grad_w{l} = zeros(size(model.weights{l}));
        num_grad_b{l} = zeros(size(model.biases{l}));
        
        for j = 1:size(model.weights{l},2)
            for k = 1:size(model.weights{l},1);
                model_p_h = model;
                model_p_h.weights{l}(k,j) = model_p_h.weights{l}(k,j) + h;
                num_grad_w{l}(k,j) = (C(model_p_h, X, y) - C(model, X, y)) / h;
            end
            
            model_p_h = model;
            model_p_h.biases{l}(j) = model_p_h.biases{l}(j) + h;
            num_grad_b{l}(j) = (C(model_p_h, X, y) - C(model, X, y)) / h;
        end
    end
end

