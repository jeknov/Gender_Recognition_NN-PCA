%% **** Created by Jekaterina Novikova  ****
%% *****************************************
%  **** PCA and Neural Network Learning ****
%  ****   for human gender recognition  ****
%  ****       from a given picture      ****
%  *****************************************

%% Initialization
clear ; close all; clc

%% Setup the NN parameters
input_layer_size  = 10;  % 10 principal features
hidden_layer_size = 25;   % 25 hidden units
num_labels = 2;          % 2 genders

%% =============== Part 1.1: Loading and Visualizing Face Data =============
%  We start the exercise by first loading and visualizing the dataset.
%  The following code will load the dataset into your environment
%
fprintf('\nLoading face dataset.\n\n');

%  Load Face dataset
load ('faces.mat')
X=X(1:500,:);
%  Display the first 100 faces in the dataset
displayData(X(1:100, :));

fprintf('Program paused. Press enter to continue.\n');
pause;


%% =========== Part 1.2: PCA on Face Data: Eigenfaces  ===================
%  Run PCA and visualize the eigenvectors which are in this case eigenfaces
%  We display the first 36 eigenfaces.
%
fprintf(['\nRunning PCA on face dataset.\n' ...
         '(this mght take a minute or two ...)\n\n']);

%  Before running PCA, it is important to first normalize X by subtracting 
%  the mean value from each feature
[X_norm, mu, sigma] = featureNormalize(X);

%  Run PCA
[U, S] = pca(X_norm);

%  Visualize the top 36 eigenvectors found
displayData(U(:, 1:100)');

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ============= Part 1.3: Dimension Reduction for Faces =================
%  Project images to the eigen space using the top k eigenvectors 
%  If you are applying a machine learning algorithm 
fprintf('\nDimension reduction for face dataset.\n\n');

K = 10;
Z = projectData(X_norm, U, K);

fprintf('The projected data Z has a size of: ')
fprintf('%d ', size(Z));

fprintf('\n\nProgram paused. Press enter to continue.\n');
pause;

%% === Part 1.4: Visualization of Faces after PCA Dimension Reduction ===
%  Project images to the eigen space using the top K eigen vectors and 
%  visualize only using those K dimensions
%  Compare to the original input, which is also displayed

fprintf('\nVisualizing the projected (reduced dimension) faces.\n\n');

K = 10;
X_rec  = recoverData(Z, U, K);

% Display normalized data
subplot(1, 2, 1);
displayData(X_norm(1:100,:));
title('Original faces');
axis square;

% Display reconstructed data from only k eigenfaces
subplot(1, 2, 2);
displayData(X_rec(1:100,:));
title('Recovered faces');
axis square;

% fprintf('Program paused. Press enter to continue.\n');
% pause;

%% ===========================END of PCA==================================

m = size(Z, 1);

% Randomly select 100 data points to display
sel_all = randperm(size(Z, 1));
sel_train=sel_all(1:300);
sel_cv=sel_all(301:400);
sel_test=sel_all(401:500);

Ztrain=Z(sel_train,:);
Ztest=Z(sel_test,:);
Zval=Z(sel_cv,:);

ytrain=y(1:300);
yval=y(301:400);
ytest=y(401:500);
% y=ytrain;

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================ Part 2: Loading Pameters ================
% Loading some pre-initialized 
% neural network parameters.

fprintf('\nLoading Saved Neural Network Parameters ...\n')

% Load the weights into variables Theta1 and Theta2
load('weights.mat');

% Load only three columns of data
sel_Theta1 = randperm(size(Theta1, 2));
sel_Theta1 = sel_Theta1(1:11);
Theta1=Theta1(:,sel_Theta1);

sel_Theta2 = randperm(size(Theta2, 1));
sel_Theta2 = sel_Theta2(1:2);
Theta2=Theta2(sel_Theta2,:);

% Unroll parameters 
nn_params = [Theta1(:) ; Theta2(:)];

%% ================ Part 3: Compute Cost (Feedforward) ================
% =================== and Implement Regularization ====================
%  Implementing the feedforward part of
%  the neural network that returns the cost only. 

fprintf('\nFeedforward Using Neural Network ...\n')

% % Weight regularization parameter (set it to 0).
% lambda = 0;
% 
% J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
%                    num_labels, X1, y, lambda);


%  Implementing the regularization of the cost.

fprintf('\nChecking Cost Function (w/ Regularization) ... \n')

% Weight regularization parameter (we set this to 1 here).
lambda = 2.8;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, Z, y, lambda);

%% ================ Part 4: Initializing Parameters ================
%  Implmenting a two layer neural network. 

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%% =============== Part 6: Implement Backpropagation with Regularization ===============
%  Implementing the regularization with the cost and gradient.

fprintf('\nChecking Backpropagation (w/ Regularization) ... \n')

%  Check gradients by running checkNNGradients
lambda = 2.8;
%checkNNGradients(lambda);

% Also output the costFunction debugging values
debug_J  = nnCostFunction(nn_params, input_layer_size, ...
                          hidden_layer_size, num_labels, Z, y, lambda);

fprintf(['\n\nCost at (fixed) debugging parameters (w/ lambda = 10): %f ' ...
         '\n(this value should be about 0.576051)\n\n'], debug_J);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 9: Learning Curve for Neural Network =============
%  Implementing the learningCurve function. 
%

lambda = 2.8;
w=size(Ztrain);

for i=1:w 
    error_train(i) = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, Ztrain(1:i,:), ytrain(1:i), lambda);
    error_val(i) = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, Zval, yval, lambda);
end

fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:w
     fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end

figure, plot(1:300, error_train, 1:300, error_val);
title('Learning curve for NN')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
axis([0 300 5 12])

error_train=error_train';
error_val=error_val';
fprintf('Program paused. Press enter to continue.\n');
pause;

%% =================== Part 7: Training NN ===================
%  Training a neural 
%  network. To train the neural network, I will now use "fmincg"
%
fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 200);

%  Try different values of lambda
lambda = 2.8;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, Z, y, lambda);
costFunctionval = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, Zval, yval, lambda);
% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
[nn_params_val, costval] = fmincg(costFunctionval, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================= Part 8: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

pred = predict(Theta1, Theta2, Z);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


%% =========== Part 10: Visualising Results =============


% %  To give an idea of the network's output, run
% %  through the examples one at the a time to see what it is predicting.
% 
% %  Randomly permute examples
% rp = randperm(m);

% [w x]=size(Zval);
% Xtest=X(401:500, :);

for i = 1:m    
     % Display 
     fprintf('\nDisplaying Example Image No. %d \n', i);
     displayData(X(i, :));
 
     pred = predict(Theta1, Theta2, Z(i,:));     
         if pred==1
             gender='male';
         else
             gender='female';
         end
     acc(i,1)=double(pred == y(i)) * 100;
     
     fprintf('\nNeural Network Prediction: %s. ', gender);
     fprintf('Accuracy is %d %%.\n', acc(i,1));
     
     % Pause
      fprintf('Program paused. Press enter to continue.\n');
      pause;
end
% save('accuracy_95proc_2.8l_200iter.mat', 'acc');
