function [w,nll] = trainMLE(X,Y,edgeStruct,inferFunc,C,w)
%
% Trains an MRF using MLE.
%
% X : nNode x nTrain matrix of training observations
% Y : nNode x nTrain matrix of training labels
% edgeStruct : edge structure
% inferFunc : inference function
% C : regularization constant (optional: def=nNode)
% w : init weights (optional: def=zeros)

% parse input
if nargin < 4
	error('USAGE: trainMLE(X,Y,edgeStruct,inferFunc)')
	return
end
[nNode,nTrain] = size(Y);
if nargin < 5
	C = nNode;
end

% edge features
Xedge = UGM_makeEdgeFeatures(X,edgeStruct.edgeEnds);

% maps
if nargin < 6
	[nodeMap,edgeMap,w] = UGM_makeCRFmaps(X,Xedge,edgeStruct,0,1);
else
	[nodeMap,edgeMap] = UGM_makeCRFmaps(X,Xedge,edgeStruct,0,1);
end

% L2 regularization
lambda = C * ones(size(w));
lambda(1) = 0;
obj = @(w)penalizedL2(w,@UGM_CRF_NLL,lambda,X,Xedge,Y',nodeMap,edgeMap,edgeStruct,inferFunc);

% Optimize
w = minFunc(obj,w);
nll = UGM_CRF_NLL(w,X,Xedge,Y',nodeMap,edgeMap,edgeStruct,inferFunc);

