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

h = sigmoid(X * theta);

n = size(theta, 1);
sum_ntheta = 0;

ntheta = theta.^2;
for i = 2:n,
    sum_ntheta = sum_ntheta + ntheta(i);
end;

J = 1/m *((-y' * log (h) - (1 - y)' * log(1 - h)) + lambda/2 *(sum_ntheta));

%find what wrong
%grad_a = zeros(size(theta));
%for i = 1 : m,
%    grad_a = grad_a + (h(i) - y(i)) * X (i,:)';
%end;
%grad_a = grad_a * 1/m;
%for j = 2 : n,
%    for i = 1 : m,
%        grad = grad + (h(i) - y(i)) * X (i,:)';
%    end;
%    grad = 1/m * (grad + lambda * theta(j));
%end;
%grad(1) = grad_a(1);

%theta 0
grad(1) = 1/m * X(:,1)'*(h - y);

%theta 1 - n
grad(2:length(grad)) = (1/m * X(:,2:size(X,2))'*(h - y)) + (lambda/m * theta(2:length(theta)));

% =============================================================

end
