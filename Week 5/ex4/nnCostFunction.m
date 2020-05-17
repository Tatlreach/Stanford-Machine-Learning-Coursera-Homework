function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
a2 = sigmoid([ones(m,1) X] * Theta1');

a3 = sigmoid([ones(m,1) a2] * Theta2');
% NOTE: check the correct dimensions here

% recode y to be of 1s, 0s
training_ans = zeros(m, num_labels);
for i=1:m,
    training_ans(i, y(i)) = 1;
end
y = training_ans;

% cost function for individual values (logistic regression?)
% J = -(1/m) * (y' .* log(a2)) + ((1-y)' .* log(1-a2)) ));

% cost func for (m by K) matrix of answers
J = -(1/m) * sum(sum( (y .* log(a3)) + ((1-y) .* log(1-a3)) ));

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

% init deltasums to 0
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
for t=1:m,
    % get data instance x
    a1 = [1 X(t,:)];
    
    % forward feed x
    z2 = a1 * Theta1';
    a2 = [1, sigmoid(z2)];
    
    z3 = a2 * Theta2';
    a3 = sigmoid(z3);
    % a3 /* output */  doesn't need a bias unit attached
    
    % get final delta = (forward_feed_ans - y);
    yt = y(t,:);
    delta_a3 = (a3 - yt);
    
    % NOTE: NO BIAS UNIT in delta_a3
    % calc delta_Theta2 = (delta_a3' * a2)
    % Theta2_sum += delta_Theta2
    Theta2_grad = Theta2_grad + (delta_a3' * a2);
    
    % backpropagate delta_a(l)
        % g'(z(l)) = z(l) .* (1-z(l));
        % delta_a(l) = remove_bias(delta_a(l+1) * Theta(l)) .* g'(z(l))
    
    % NOTE: NO BIAS UNIT  in z2, a3, delta_a3
    gz_prime = sigmoidGradient( z2 );
    
    % NOTE: delta_a3 does NOT have BIAS
    % NOTE: result(delta_a2) has BIAS because Theta2 does
    delta_a2 = (delta_a3 * Theta2);
    %remove bias column
    delta_a2 = delta_a2(2:end);
    delta_a2 = delta_a2 .* gz_prime;
    
    % NOTE  (K x m) * (m x a2_sz+1) = (K x a2_sz+1) = size(Theta2)
    
    % calc delta_Theta1(delta_a2' * a1)
    % not this?? % Theta1_grad = Theta1_grad + delta1;
    Theta1_grad = Theta1_grad + (delta_a2' * a1);
    % NOTE  (a2_sz x m) * (m x a1_sz+1) = (a2_sz x a1_sz+1) = size(Theta2)
end
% average delta_theta sums
Theta1_grad = Theta1_grad/m;
Theta2_grad = Theta2_grad/m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% remove 1st column
%Theta1_reg = Theta1(:, 2:size(Theta1,2));
%Theta2_reg = Theta2(:, 2:size(Theta2,2));
% NOTE: size(mtrx,2) is col size

% regularize the theta vals
Theta1_reg = Theta1;
Theta2_reg = Theta2;

% set first column to 0
% don't regularize bias component
Theta1_reg(:, 1) = zeros(size(Theta1,1), 1);
Theta2_reg(:, 1) = zeros(size(Theta2,1), 1);

% add regularized component of delta thetas
Theta1_grad = Theta1_grad + (lambda/m) * Theta1_reg;
Theta2_grad = Theta2_grad + (lambda/m) * Theta2_reg;

% unroll delta_theta sums into a vector
grad = [Theta1_grad(:); Theta2_grad(:)];

% square thetas
Theta1_reg = Theta1_reg .^ 2;
Theta2_reg = Theta2_reg .^ 2;

J = J + (lambda/(2*m)) * (sum(sum(Theta1_reg)) + sum(sum(Theta2_reg)));
















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
