function [w,nll] = trainMLE(X,Y,edgeStruct,nodeMap,edgeMap,inferFunc,C,w)
%
% Trains an MRF using MLE.
%
% X : nNode x nTrain matrix of training observations
% Y : nNode x nTrain matrix of training labels
% edgeStruct : edge structure
% nodeMap : maps nodes to params
% edgeMap : maps edges to params
% inferFunc : inference function
% C : regularization constant (optional: def=nNodeY)
% w : init weights (optional: def=zeros)

% parse input
if nargin < 6
	error('USAGE: trainMLE(X,Y,edgeStruct,nodeMap,edgeMap,inferFunc)')
	return
end
nTrain = size(X,2);
if nargin < 7
	C = nTrain;
end
if nargin < 8
	w = zeros(max(max(nodeMap(:)),max(edgeMap(:))),1);
end

% compute sufficient statistics
suffStat = UGM_MRF_computeSuffStat([Y ; X]',nodeMap,edgeMap,edgeStruct);

% L2 regularization
lambda = C * ones(size(w));
lambda(1) = 0;
obj = @(w)penalizedL2(w,@UGM_MRF_NLL,lambda,nTrain,suffStat,nodeMap,edgeMap,edgeStruct,inferFunc);

% Optimize
w = minFunc(obj,w);
nll = UGM_MRF_NLL(w,nTrain,suffStat,nodeMap,edgeMap,edgeStruct,inferFunc);

