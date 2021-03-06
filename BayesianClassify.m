
function [classLabels] = BayesianClassify(model, testData)
% function [classLabels] = BayesianClassify(model, testData)
%
% Gives the class labels of testData according to the given model
%
% INPUT:
%
% model    : k x 2 cell, k is num of classes.
%            Each row i is {muHat(mean_vector)_i, C(covariance_matrix)_i}
%
% testData     : m x n matrix, m is num of examples & n is
% number of dimensions.
%
% OUTPUT:
%
% classLabels: m x 1 matrix, labels of testData, 1 for class 1, ... , k for
% class k.
%
% See Also : BuildBaysianModel.m
%

m = size(testData, 1); % number of examples
n = size(testData, 2); % number of feature dimension
k = size(model, 1); % number of classes

classLabels  = zeros(m,1);
for l = 1:m
p_i = 0;
maxlab = 0;    
    for i=1:k
    p = (1/((2*pi)^(-n/2) * det(cell2mat(model(i,2)))^0.5)) * exp(-0.5*(bsxfun(@minus,testData(l,:),cell2mat(model(i,1))))*inv(cell2mat(model(i,2)))*(bsxfun(@minus,testData(l,:),cell2mat(model(i,1))))');
    
    if(p_i < p)
                
        maxlab = i;
        p_i = p;
        
    end
    end
    
    classLabels(l,1) = maxlab;
end

% Complete the function
end