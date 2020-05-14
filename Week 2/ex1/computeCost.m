function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% theta = current feature scalar estimation

% X = whole dataset (m x n)
% y = given answers (m x 1)
% theta = current scalars (n x 1)
% hx = scalars x data = current estimation  (m x 1)

% disp(X)
% disp(theta)
% disp(y)
% J = (1 / m) * (hx - y)^2
%disp(theta)
hx = X * theta;
% J = hx - y
% J = J .* J
% J = sum(J)
% J =  J / (2*m)

%J = 1/(2*m) * (sum( (hx-y).^2 ));
J = 1/(2*m) * (sum( (hx-y).^2 ));
% J = J/2;

% =========================================================================

end
