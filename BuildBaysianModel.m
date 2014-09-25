
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


%%

%To DO:

%Variance coupling. Try wishart distribution thing. Should be good.
%%
m = size(trainData, 1); % number of training examples
n = size(trainData, 2) - 1; % number of feature dimension
k = length(unique(trainData(:, end))); % number of classes
classes = unique(trainData(:, end));
model = cell(k, 2);
class_matrices = zeros(m,n,k);
C = zeros(n,n);
counts = ones(1,k);
%%For the combination of covariance matrices, we'll use the pooled
%%covariance, mean and Wishart distribution estimates for the model.
%%Whichever works the best. 
for i = 1:m

        class_matrices(counts(1,trainData(i,n+1)),:,trainData(i,n+1)) = trainData(i,1:n);
        counts(1,trainData(i,n+1)) = counts(1,trainData(i,n+1)) + 1;
end
%% Avegaing out the Cov Matrices %%
if(caseNumber == 1)
    
    for i = 1:k
    C = C + (1/k)*(cov(class_matrices(:,:,i)));
    end
    
    for i = 1:k
        model(i,2) = mat2cell(C);%cov(class_matrices(:,:,i);
    end
end
    
if(caseNumber == 2)
    
%Splitting the data into class matrices, so that we can find the mean and
%variance for each class for the multinomial distribution.

%cov_matrices = zeros(n,n,k);
%mean_matrices = zeros(n,k);
%determinants = zeros(k);
    for i = 1:k
        model(i,2) = {cov(class_matrices(:,:,i))};
    end
end


if(caseNumber == 3)
    
    varmat = zeros(1,k);
    for i = 1:k
        varmat(1,i) = var(diag(class_matrices(:,:,i)));
    end
    
    varz = 0;
    
    for i = 1:k
        varz = varz + (size(class_matrices(:,:,i),1)-1)*varmat(1,i);
    end
    
    varz = varz/(m-k);
    
    for i = 1:k
        model(i,2) = {varz*eye(n)};
    end
        
end



if(caseNumber == 4)
    for i = 1:k
        C = C + (1/k)*diag(diag((cov(class_matrices(:,:,i)))));
    end
    
    for i = 1:k
        model(i,2) = {C};%cov(class_matrices(:,:,i);
    end
        
end

    
if(caseNumber == 5)
    
%Splitting the data into class matrices, so that we can find the mean and
%variance for each class for the multinomial distribution.

%cov_matrices = zeros(n,n,k);
%mean_matrices = zeros(n,k);
%determinants = zeros(k);
    for i = 1:k
        model(i,2) = {diag(diag(cov(class_matrices(:,:,i))))};
    end
end


for i = 1:k
    model(i,1) = mat2cell(mean(class_matrices(1:counts(1,trainData(i,n+1))-1,:,i)));
    %determinants(i) = det(cov_matrices(:,:,i));
end
% Complete the function


end
