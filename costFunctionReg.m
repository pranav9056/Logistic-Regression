function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
thetaX = X * theta;
predicted = sigmoid(thetaX);
J = -1/m * sum(y .* log(predicted) + (1-y) .* log(1-predicted));
J = J + (lambda/(2*m))* sum(theta(2:end) .^ 2);
grad = sum((predicted - y) .* X);
grad = grad';
grad = grad ./ m;
gradRegularizer = lambda/m .* theta;
gradRegularizer(1) = 0;
grad = grad + gradRegularizer;
end
