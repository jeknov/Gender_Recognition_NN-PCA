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
% for the 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
K=3; %Number of layers in the net
         
% The following variables should be returned
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


% Part 1: Feedforward the neural network and return the cost in the
%         variable J.

X = [ones(m, 1) X]; %Add bias to input layer

z2=X*Theta1'; %Calculates z for hidden layer
a2_no_bias=sigmoid(z2); %Calculates hidden layer's values
a2=[ones(m, 1) a2_no_bias]; %Add bias to hidden layer

z3=a2*Theta2'; %Calculates z for output layer
a3=sigmoid(z3); %Calculates output layer's values

e = eye(num_labels);
y1=e(y,:); %Re-write the Y vaues by vectors with 1's on the right place

%Vectorized regularized cost function
J=1/m * sum(sum(-1 * y1 .* log(a3)-(1-y1) .* log(1-a3)))+(lambda/(2*m)) * [(sum(sum(Theta1(:,2:end).^2))) + (sum(sum(Theta2(:,2:end).^2)))]; 

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. The partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively, should be returned. After that,
%         implement regularization with the cost function and gradients.
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. I map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.

delta3=a3-y1; % Where y1 and a3 are 5000x10 matrices

Theta22=Theta2(:,2:end);
delta2=delta3*Theta22.*sigmoidGradient(z2);

Delta_2=delta3'*a2;
Delta_1=delta2'*X;

%Theta1_grad=1/m*Delta_1; %correct if no regularization
Delta_1=1/m*Delta_1;

%szD2=size(Delta_2)
%szReg=size(lambda/m*Theta2)

%Theta2_grad=1/m*Delta_2; %correct if no regularization
Delta_2=1/m*Delta_2;

Theta1(:,1)=0; Theta2(:,1)=0;
Delta_1 = Delta_1+lambda/m*Theta1;
Delta_2 = Delta_2+lambda/m*Theta2;

Theta1_grad=Delta_1;
Theta2_grad=Delta_2;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
