
function [model] = BuildBaysianModel(trainData, crossValidationData, caseNumber)
% function [model] = BuildBaysianModel(trainData, crossValidationData, caseNumber)
%
% Builds Bayesian model using the given training data and cross validation
% data (optional) for the given case number.
%
% INPUT:
%
% trainData     : m x n+1 matrix, m is num of examples & n is number of
% dimensions. n+1 th column is for class labels (1 -- for class 1, ... k --
% for class k).
%
% crossValidationData     : (Optional) m x n+1 matrix, m is num of examples & n is
% number of dimensions. n+1 th column is for class labels (1 -- for class
% 1, ... , k -- for class k).
%
% caseNumber: 1 -- Bayes with Covariance same for all classes
%             2 -- Bayes with Covariance different for all classes
%             3 -- Naive Bayes with C = \sigma^2*I
%             4 -- Naive Bayes with C same for all
%             5 -- Naive Bayes with C different for all
%
% OUTPUT:
% model    : k x 2 cell, k is num of classes.
%            Each row i is {muHat(mean_vector)_i, C(covariance_matrix)_i}
%
% See Also : BayesianClassify.m
%

m = size(trainData, 1); % number of training examples
n = size(trainData, 2) - 1; % number of feature dimension
k = length(unique(trainData(:, end))); % number of classes
classes = unique(trainData(:, end));
model = cell(k, 2);
class_matrices = zeros(m,n,k);

%%For the combination of covariance matrices, we'll use the pooled
%%covariance, mean and Wishart distribution estimates for the model.
%%Whichever works the best. 


if(caseNumber == 1)
    C = cov(trainData(:,1:n));
    for i = 1:k
        model(k,2) = C;%cov(class_matrices(:,:,i);
    end
end
    
if(caseNumber == 2)
    
%Splitting the data into class matrices, so that we can find the mean and
%variance for each class for the multinomial distribution.
    for i = 1:m
        class_matrices(:,:,find(classes,trainData(i,-1)) = trainData(trainData(i,1:n);
    end

%cov_matrices = zeros(n,n,k);
%mean_matrices = zeros(n,k);
%determinants = zeros(k);
    for i = 1:k
        model(k,2) = cov(class_matrices(:,:,i);
    end
end


for i = 1:k
    model(k,1) = mean(class_matrices(:,:,i));
    %determinants(i) = det(cov_matrices(:,:,i));
end





% Complete the function


end
