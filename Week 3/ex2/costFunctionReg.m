function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Cost Function Regularized
z = X * theta;
gz = sigmoid(z);
J = -(1/m) * (y' * log(gz) + (1-y')*log(1-gz));


% remove column 1 from theta!
theta_2 = theta;
theta_2(1) = 0;
J = J + lambda/(2*m) * (theta_2' * theta_2);

% Gradient Descent Regularized
% grad = theta - alpha/m * (X' * (gz - y))
% grad = (1/m) * (X' * (gz-y))
% grad = Θj(1-lambda*alpha/m) + (1/m)*(X' * (gz - y))
% grad = Θj - Θj*lambda*alpha/m + (1/m)*(X' * (gz - y))
grad = theta_2*(lambda/m) + (1/m)*(X' * (gz - y));

% =============================================================

end
