function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
thetaX = X * theta;
predicted = sigmoid(thetaX);
J = -1/m * sum(y .* log(predicted) + (1-y) .* log(1-predicted));
grad = sum((predicted - y) .* X);
grad = grad';
grad = grad ./ m;
end
