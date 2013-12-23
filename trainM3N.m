function [w,loss] = trainM3N(X,Y,edgeStruct,inferFunc,C,maxIter,w)
%
% Trains an MRF using max-margin formulation.
%
% X : nNode x nTrain matrix of training observations
% Y : nNode x nTrain matrix of training labels
% edgeStruct : edge structure
% inferFunc : inference function
% C : regularization constant (optional: def=nNode)
% maxIter : max. number of iterations of SGD (optional: def=10)
% w : init weights (optional: def=zeros)

% parse input
if nargin < 4
	error('USAGE: trainM3N(X,Y,edgeStruct,inferFunc)')
	return
end
[nNode,nTrain] = size(Y);
if nargin < 5
	C = nNode;
end
if nargin < 6
	maxIter = 10;
end

% edge features
Xedge = UGM_makeEdgeFeatures(X,edgeStruct.edgeEnds);

% maps
if nargin < 7
	[nodeMap,edgeMap,w] = UGM_makeCRFmaps(X,Xedge,edgeStruct,0,1);
else
	[nodeMap,edgeMap] = UGM_makeCRFmaps(X,Xedge,edgeStruct,0,1);
end

% SGD
stepSize = 1e-4;
fAvg = 0;
for iter = 1:maxIter*nTrain
	% Compute M3N objective and subgradient for random training example
	i = ceil(rand*nTrain);
	[f,g] = UGM_M3N_Obj(w,X(i,:,:),Xedge(i,:,:),Y(:,i)',nodeMap,edgeMap,edgeStruct,inferFunc);
	
	% L2 regularization
	f = f + 0.5*C*w'*w;
	g = g + C*w;
	
	% Update estimate of function value and parameters
	fAvg = (1/iter)*f + ((iter-1)/iter)*fAvg;
	w = w - stepSize*g;
	
	fprintf('Iter = %d of %d (ex %d: f = %f, fAvg = %f)\n',iter,maxIter*nTrain,i,f,fAvg);
end

loss = fAvg;

