function [w,loss] = trainM3N(examples,decodeFunc,C,maxIter,w)
%
% Trains an MRF using max-margin formulation.
%
% examples : cell array of examples
% decodeFunc : decoder function
% C : regularization constant or nParam x 1 vector (optional: def=nNode of first example)
% maxIter : max. number of iterations of SGD (optional: def=10)
% w : init weights (optional: def=zeros)

% parse input
assert(nargin >= 2, 'USAGE: trainM3N(examples,inferFunc)')
nTrain = length(examples);
if nargin < 3
	C = examples{1}.nNode;
end
if nargin < 4
	maxIter = 10;
end
if nargin < 5
	nParam = max(max(examples{1}.nodeMap(:)),max(examples{1}.edgeMap(:)));
	w = zeros(nParam,1);
end

% SGD
stepSize = 1e-4;
fAvg = 0;
for iter = 1:maxIter*nTrain
	% Compute M3N objective and subgradient for random training example
	i = ceil(rand*nTrain);
	ex = examples{i};
	[f,g] = UGM_M3N_Obj(w,ex.Xnode,ex.Xedge,ex.Y',ex.nodeMap,ex.edgeMap,ex.edgeStruct,decodeFunc);
	
	% L2 regularization
	f = f + 0.5 * (C.*w)' * w;
	g = g + C.*w;
	
	% Update estimate of function value and parameters
	fAvg = (1/iter)*f + ((iter-1)/iter)*fAvg;
	w = w - stepSize*g;
	
	fprintf('Iter = %d of %d (ex %d: f = %f, fAvg = %f)\n',iter,maxIter*nTrain,i,f,fAvg);
end

loss = fAvg;

