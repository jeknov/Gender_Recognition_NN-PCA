function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

% You need to return the following variables correctly.
Z = zeros(size(X, 1), K);
n = size(X);

% ======================================================================
% Compute the projection of the data using only the top K 
% eigenvectors in U (first K columns). 

U_reduce = U(:, 1:K); % calculates first K value of reduced U
Z = X*U_reduce; % calculates projection Z
% =============================================================

end
