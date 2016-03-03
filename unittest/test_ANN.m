classdef test_ANN < matlab.unittest.TestCase

    methods( Test, Access=public )
    
        function test_feedforward(testCase)

            % 4 layer neural network
            model = initialize_model( [3 2 2 1], {[ 1 2 ; 2 1 ; 1 1 ], [-1 1 ; -2 2], [2 ; 3]}, {[1 ; -1], [-2 ; -2], 1} );

            % Test with one sample
            inputs1 = [ 1 2 3 ];
            z = feedforward( model, inputs1 );
            assertTrue( testCase,  abs(z - 0.9610) < 1.0e-4 );

            % Test with two samples
            inputs2 = [ 1 2 3 ; 1 2 3 ];
            z = feedforward( model, inputs2 );
            assertEqual( testCase, (abs(z - 0.9610) < [1.0e-4 ; 1.0e-4]), [true ; true] );    % not complete

            % Test with double and single data types
            model = initialize_model( double([3 2 2 1]) );  % Defaults to double
            inputs3 = double( [ 1 2 3 ] );
            z = feedforward( model, inputs3 );          
            assertTrue( testCase, isa( z, 'double' ) );
            assertTrue( testCase, isa( model.weights{1}, 'double' ) );
            
            model = initialize_model( single([3 2 2 1]), single([]) );
            inputs4 = single( [ 1 2 3 ] );
            z = feedforward( model, inputs4 );  % force single with empty single array
            verifyTrue( testCase, isa( z, 'single' ) );
            for i=1:numel(model.weights); assertTrue( testCase, isa(model.weights{i}, 'single') ); end;
            
            % Test with GPU array
            model = initialize_model( single([3 2 2 1]), gpuArray(double([])) );
            inputs5 = single( [ 1 2 3 ] );
            z = feedforward( model, inputs5 );
            assertTrue( testCase, isa( z, 'gpuArray' ) );
            for i=1:numel(model.weights); assertTrue( testCase, isa(model.weights{i}, 'gpuArray') ); end;
            
            % Test single/double on GPU array
            model = initialize_model( single([3 2 2 1]), gpuArray(single([])) );
            inputs6 = single( [ 1 2 3 ] );
            z = feedforward( model, inputs6 );
            assertTrue( testCase, isa( z, 'gpuArray' ) );
            for i=1:numel(model.weights); assertTrue( testCase, strcmpi( classUnderlying(model.weights{i}), 'single' ) ); end;
            
            model = initialize_model( double([3 2 2 1]), gpuArray(double([])) );
            inputs7 = single( [ 1 2 3 ] );
            z = feedforward( model, inputs7 );
            assertTrue( testCase, isa( z, 'gpuArray' ) );
            for i=1:numel(model.weights); assertTrue( testCase, strcmpi( classUnderlying(model.weights{i}), 'double' ) ); end;
            
            % And gate
            model_andgate = initialize_model( [2 1], {[0.5 ; 0.5]}, {-0.6} );
            predict = feedforward( model_andgate, [1 1 ; 1 0 ; 0 1 ; 0 0] );
            assertTrue( testCase, all( (predict>0.5) == [1 0 0 0]' ) );
        end

        
        function test_sigmoid(testCase)
            % sanity and edge cases
            assertTrue( testCase, sigmoid(inf) == 1 );
            assertTrue( testCase, sigmoid(-inf) == 0 );
            assertTrue( testCase, sigmoid(0) == 0.5 );
            assertTrue( testCase, sigmoid(999999999) > 1.0e-100 );
            assertTrue( testCase, sigmoid(-999999999) == 0 );
            
            % single value case
            assertTrue( testCase, numel(sigmoid(0)) == 1 );
            
            % array of values case
            assertEqual( testCase, size(sigmoid([-1 ; 0 ; 1])), [3 1] );
            assertEqual( testCase, sigmoid([-2 ; 0 ; 2]), [sigmoid(-2) ; sigmoid(0) ; sigmoid(2) ] );
            
            % single, double, gpuArray
            assertTrue( testCase, isa(sigmoid( single(0)   ), 'single') );
            assertTrue( testCase, isa(sigmoid( double(0)   ), 'double') );
            assertTrue( testCase, isa(sigmoid( gpuArray(0) ), 'gpuArray') );
        end
        
                
        
        function test_backprop(testCase)
            % Basic 2 features hand computed ANN, 1 sample
            model = initialize_model( [2 2 1], {[3 1 ; 2 4], [1 ; -2]}, {[1 ; 2], 3} );
            [grad_w, grad_b] = backprop( model, [2 3], 0 );
            assertTrue( testCase, test_ANN.compareFloat( grad_b{2}, 9.247814e-02, 1.0e-7 ) );
            assertTrue( testCase, test_ANN.compareFloat( grad_w{2}, [9.247793e-02 ;  9.247813e-02], 1.0e-7 ) );
            assertTrue( testCase, test_ANN.compareFloat( grad_b{1}, [2.090302e-07 ; -2.081409e-08], 1.0e-7 ) );
            assertTrue( testCase, test_ANN.compareFloat( grad_w{1}, [4.180604e-07 -4.162818e-08 ; 6.270906e-07 -6.244227e-08], 1.0e-7 ) );

            % Check weights against numerical gradients
            [num_grad_w, num_grad_b] = numericalGradients(model, [2 3], 0 );
            assertTrue( testCase, test_ANN.compareFloat( grad_w{1}, num_grad_w{1}, 1.0e-3 ) );
            assertTrue( testCase, test_ANN.compareFloat( grad_w{2}, num_grad_w{2}, 1.0e-3 ) );
            % Check biases against numerical gradients
            assertTrue( testCase, test_ANN.compareFloat( grad_b{1}, num_grad_b{1}, 1.0e-3 ) );
            assertTrue( testCase, test_ANN.compareFloat( grad_b{2}, num_grad_b{2}, 1.0e-3 ) );
            
            % Sanity check with 3 samples
            model = initialize_model( [4 3 2] );
            [grad_w, grad_b] = backprop( model, [1 2 3 4;2 3 4 5;3 4 5 6;4 5 6 7;5 6 7 8], [1 0;0 1;1 1;1 0;0 1] );
            assertTrue( testCase, all(size(grad_w{1}) == [4 3]) );
            assertTrue( testCase, all(size(grad_w{2}) == [3 2]) );
            assertTrue( testCase, all(size(grad_b{1}) == [1 3]) );
            assertTrue( testCase, all(size(grad_b{2}) == [1 2]) );
% 
%             
%             assertTrue( testCase, false ); % TOOD
%             
%             % Test single, and gpuArray
%             assertTrue( testCase, false );  % TODO
        end
        
        
        function test_mini_batch(testCase)
            model = initialize_model( [2 2 1], {[3 1 ; 2 4], [1 ; -2]}, {[1 ; 2], 3} );
            updated_model = update_mini_batch( model, [2 3 ; 2 3], [0 ; 0], 0.1 );
            assertTrue( testCase, test_ANN.compareFloat( updated_model.biases{2}, 2.990752183586165, 1.0e-7 ) );
            assertTrue( testCase, test_ANN.compareFloat( updated_model.weights{2}, [0.990752207000000 ; -2.009247813000000], 1.0e-7 ) );
            assertTrue( testCase, test_ANN.compareFloat( updated_model.biases{1}, [0.999999790969831 ; 2.000000020814090], 1.0e-6 ) );
            assertTrue( testCase, test_ANN.compareFloat( updated_model.weights{1}, [2.999999958199600 1.000000004162818 ; 1.999999937290940 4.000000006244222], 1.0e-7 ) );

        end
        
        function test_SGD(testCase)
            model = initialize_model( [2 2 1], {[3 1 ; 2 4], [1 ; -2]}, {[1 ; 2], 3} );
            hyper_parameters = initialize_hyper_parameters();
            hyper_parameters.mini_batch_size = 5;
            updated_model = SGD( model, hyper_parameters, repmat([2 3], [20 1]), zeros(20,1) );
            assertTrue( testCase, all( feedforward(updated_model, repmat([2 3], [20 1])) < 0.5 ) );      % Verify we learned to predictict all 0's on this contrived case.
            
            
            model = initialize_model( [2 2 2], {[3 1 ; 2 4], [1 1 ; 1 1]}, {[1 ; 2], [1 ; 2]} );
            updated_model = SGD( model, hyper_parameters, repmat([2 3], [20 1]), zeros(20,2) );
            assertTrue( testCase, all( size(model.weights{1}) == size(updated_model.weights{1}) ) );    % Sanity check
            assertTrue( testCase, all( size(model.biases{1})  == size(updated_model.biases{1})  ) );
            
            % Check with odd batch sizes
            updated_model = SGD( model, hyper_parameters, repmat([2 3], [200 1]), zeros(200,2) );       % Sanity check
            assertTrue( testCase, all( size(model.weights{1}) == size(updated_model.weights{1}) ) );    % Sanity check
            assertTrue( testCase, all( size(model.biases{1})  == size(updated_model.biases{1})  ) );
            
        end
        
    end
    
    methods(Static)
        
        function [isequal] = compareFloat( scalar_or_matrix, comparison, precision )
            isequal = all( abs(scalar_or_matrix(:) - comparison(:)) < precision );
        end
        
    end

end