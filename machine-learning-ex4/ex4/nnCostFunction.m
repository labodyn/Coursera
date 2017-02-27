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
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
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


% X has size nobs x 400
% Theta1 has size 25 x 401
% Theta2 has size 10 x 26

X0 = [ones(m, 1) X];
X_1 = sigmoid(X0*Theta1');
X_1a = [ones(m, 1) X_1];
h = sigmoid(X_1a*Theta2');

% h has size nobs x 10 
% y has size nobs x 1
% y2 has size nobs x 10

y2 = zeros(m,num_labels);

for i = 1:m
	y2(i,:) = (y(i) ==  (1:num_labels));
endfor

size(Theta1); % 25 401
size(Theta2); % 4 5 10 26 
size(X); %16 3 5000 401
size(X_1); %16 5 5000 26
size(y); %16 1 5000 1
size(y2); %16 4 5000 10
size(h); %16 4 5000 10

J = 1/m*sum(diag( -y2'*log(h) - (1-y2')*log(1-h) )) + lambda/(2*m)*( sum(diag(Theta1'*Theta1)) + sum(diag(Theta2'*Theta2)) - Theta1(:,1)'*Theta1(:,1) - Theta2(:,1)'*Theta2(:,1) );

%second_term = theta*lambda/m;
%second_term(1) = 0;

%grad = 1/m*X'*(h-y) + second_term;

D2 = 0;
D3 = 0;

for t=1:m
	%forward
	a1(t,:) = X(t,:); % 5000 400
	z2(t,:) = [1 a1(t,:)]*Theta1'; % 5000 25
	a2(t,:) = sigmoid(z2(t,:)); % 5000 25
	z3(t,:) = [1 a2(t,:)]*Theta2'; % 5000 10
	a3(t,:) = sigmoid(z3(t,:)); % 5000 10
	%backward
	d3(t,:) = a3(t,:)-y2(t,:); % 5000 10
	d2(t,:) = ( d3(t,:)*Theta2(:,2:end) ) .* sigmoidGradient(z2(t,:)); % 5000 25
	%multiply
	D3 += d3(t,:)'*[1 a2(t,:)]; % 10 26
	D2 += d2(t,:)'*[1 a1(t,:)]; % 25 401

%remove first column corresponding to bias term
Theta1_n = Theta1;
Theta2_n = Theta2;
Theta1_n(:,1) = zeros(size(Theta1,1),1);
Theta2_n(:,1) = zeros(size(Theta2,1),1);


Theta1_grad = 1/m*D2 + lambda/m*Theta1_n;
Theta2_grad = 1/m*D3 + lambda/m*Theta2_n;


% Theta1 has size 25 x 401
% Theta2 has size 10 x 26







% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
