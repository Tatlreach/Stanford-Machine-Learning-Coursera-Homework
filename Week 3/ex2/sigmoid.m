function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).


g = -1 * z;

e = 2.71828182845904523;

g = 1 + (e .^ g);  %also exp(g)
g = 1 ./ g;


% =============================================================

end
