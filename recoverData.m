function X_rec = recoverData(Z, U, K)
%RECOVERDATA Recovers an approximation of the original data when using the 
%projected data
%   X_rec = RECOVERDATA(Z, U, K) recovers an approximation the 
%   original data that has been reduced to K dimensions. It returns the
%   approximate reconstruction in X_rec.
%

% You need to return the following variables correctly.
X_rec = zeros(size(Z, 1), size(U, 1));

% =====================================================================
% Compute the approximation of the data by projecting back
% onto the original space using the top K eigenvectors in U.
%

U_reduced = U(:, 1:K); % calculates first K value of reduced U
X_rec = Z*U_reduced'; % calculates approximate recovered data X

% =============================================================

end
