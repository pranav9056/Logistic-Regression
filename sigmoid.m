function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

g = exp(-z);
g = 1 + g;
g = 1 ./ g;


end
