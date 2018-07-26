function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));  % 25 x 401
Theta2_grad = zeros(size(Theta2));  % 10 x 26

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.0
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Recode y

y_recode = zeros(m, num_labels);
for i=1:m,
  y_recode(i, y(i))=1;
end
y = y_recode;   % 5000 x 10


% Part 1 - Cost computation

a_1 = [ones(m, 1) X];   % 5000 x 401
 
a_2 = sigmoid(a_1*Theta1'); % 5000 x 25
 
a_2 = [ones(m, 1) a_2];      % 5000 x 26
a_3 = sigmoid(a_2*Theta2');     % 5000 x 10

J = (1/m) * sum ( sum ( (-y) .* log(a_3) - (1-y) .* log(1-a_3) )) + ...
        + lambda/(2*m) * (sum(sum(Theta1(:, 2:end).^2)) + sum(sum(Theta2(:, 2:end).^2)));


% Part 2 - Gradient computation

delta2 = Theta2_grad;   % 10 x 26
delta1 = Theta1_grad;   % 25 x 401

for i = 1:m

    sigma3 = a_3(i, :) - y(i, :);   % 1 x 10
    sigma2 = (sigma3 * Theta2) .* (a_2(i, :) .* (1 - a_2(i, :)));    % 1 x 26 = (1x10)x(10x26)

    delta2 = delta2 + sigma3'*a_2(i, :);    % 10 x 26 = (1x10)'x(1x26)
    delta1 = delta1 + (sigma2(2:end))'*a_1(i, :);    % 25 x 401 = (1x25)'x(1x401) - skipping sigma(0)
end


Theta2_grad = 1/m * delta2;
Theta1_grad = 1/m * delta1;

Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda/m * Theta2(:, 2:end);
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda/m * Theta1(:, 2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
