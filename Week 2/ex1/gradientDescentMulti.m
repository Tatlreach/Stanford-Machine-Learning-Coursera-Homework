function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    %   theta = theta - (alpha/m) * (X' * (hx-y))

    % disp(theta);
    % disp(X);
    % hx = zeros(m, size(X, 2));
    hx = (X * theta);
    theta = theta - (alpha/m) * (X' * (hx-y));
    
    % alternate answers
    % theta = theta - alpha * (1/m) * ((hx - y)' * X)'; 
    %theta = theta - alpha * (1/m) * (((X*theta) - y)' * X)'; % Vectorized  
    
    % disp(theta);





    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
    %disp(J_history(iter));
end
    % disp(theta);
end
