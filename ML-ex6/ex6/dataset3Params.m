function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
%C = 1;
%sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


%set trying values for C and sigma
C_vec = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_vec = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];


% error_mx stores prediction error for each combination of C and sigma
% row indexes represent C, column indexes represent sigma
error_mx = zeros(length(C_vec), length(sigma_vec));


for C = 1:length(C_vec),
    for sigma = 1:length(sigma_vec),

        % train SVM model 
        model= svmTrain(X, y, C_vec(C), @(x1, x2) gaussianKernel(x1, x2, sigma_vec(sigma)));
        
        % predict using valid-set
        predictions = svmPredict(model, Xval);
        
        % compute the prediction error
        error_mx(C,sigma) = mean(double(predictions ~= yval));
    end;
end;


% find minimum error values and index in error_mx
% error_mx(:) select all elements as a column vector
% using max/min on matrix to return index looks a bit weird :-/
[values,index] = min (error_mx(:));

% looking for subscripts (index) that have minimun error using ind2sub
[r, c] = ind2sub( size(error_mx) , index); % "ind" o trong ham duoc lay theo index dua vao column vector
% r for index of C
% c for index of sigma

C = C_vec(r);
sigma = sigma_vec(c);

% =========================================================================

end
