
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

% Complete the function
% SEX SEX SEX SEX
end