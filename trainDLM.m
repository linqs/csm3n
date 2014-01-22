function [w, fAvg] = trainDLM(examples, decodeFunc, C, options, w)
%
% Trains an MRF using direct loss minimization.
%
% examples : cell array of examples
% decodeFunc : decoder function
% C : regularization constant or nParam x 1 vector (optional: def=nNode of first example)
% options : optional struct of optimization options for SGD:
% 			maxIter : iterations of SGD (def: 10*length(examples))
% 			stepSize : SGD step size (def: 1e-4)
% 			verbose : verbose mode (def: 0)
% w : init weights (optional: def=zeros)

% parse input
assert(nargin >= 2, 'USAGE: trainM3N(examples,decodeFunc)')
if nargin < 3
	C = examples{1}.nNode;
end
if nargin < 4 || ~isstruct(options)
	options = struct();
end
if ~isfield(options,'maxIter')
	options.maxIter = 10 * length(examples);
end
if ~isfield(options,'stepSize')
	options.stepSize = 1;
end
if ~isfield(options,'verbose')
	options.verbose = 0;
end
if nargin < 5
	nParam = max(examples{1}.edgeMap(:));
	w = zeros(nParam,1);
end

% SGD
objFun = @(x,ex,t) dlmObj(x,ex,t,decodeFunc,C);
[w,fAvg] = sgd(examples,objFun,w,options);

