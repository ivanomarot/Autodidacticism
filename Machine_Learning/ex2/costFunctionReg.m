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



for i=1:m
    h_x=sigmoid(X(i,:)*theta);
    J=J+(-1*y(i)*log(h_x))-( (1-y(i))*log(1-h_x) );
end

reg=0;

for j=2:size(theta,1)
    reg=reg+((lambda/(2*m))*(theta(j)*theta(j)));
end

J=J*(1/m)+reg;

for i=1:m
    h_x=sigmoid(X(i,:)*theta);
    grad(1)=grad(1)+((1/m)*(h_x-y(i))*X(i,1));
end 

for j=2:size(theta,1)
    for i=1:m
        h_x=sigmoid(X(i,:)*theta);
        grad(j)=grad(j)+((1/m)*(h_x-y(i))*X(i,j));
    end
    grad(j)=grad(j)+((lambda/m)*theta(j));
end



% =============================================================

end
