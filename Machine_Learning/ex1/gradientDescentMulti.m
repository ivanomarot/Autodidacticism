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


  sum_grad = zeros(size(X, 2)+1, 1);
  temp_theta = zeros(size(X, 2)+1, 1);
  for i=1:m
    h_x=0;
    for j=1:size(X, 2)
        h_x=h_x+(theta(j,1)*X(i,j));
    end
    for j=1:size(X, 2)
        sum_grad(j,1)=sum_grad(j,1)+((h_x-y(i))*X(i,j));
    end
  end
  
  for j=1:size(X, 2)
      temp_theta(j,1)=theta(j,1)-((alpha/m)*sum_grad(j,1));
  end
  for j=1:size(X, 2)
      theta(j,1)=temp_theta(j,1);
  end


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
