function [w, nll] = trainMLE(examples, inferFunc, C, options, w)
%
% Trains an MRF using MLE.
%
% examples : cell array of examples
% inferFunc : inference function (0: use pseudolikelihood)
% C : optional regularization constant or vector (def: 1)
% options : optional struct of optimization options for SGD:
% 			maxIter : iterations of SGD (def: 100*length(examples))
% 			stepSize : SGD step size (def: 1e-4)
% 			verbose : verbose mode (def: 0)
% w : init weights (optional: def=zeros)

% parse input
assert(nargin >= 2, 'USAGE: trainM3N(examples,inferFunc)')
usePL = ~isa(inferFunc,'function_handle');
if nargin < 3
	C = 1;
end
if nargin < 4 || ~isstruct(options)
	options = struct();
end
if ~isfield(options,'maxIter')
	options.maxIter = 100 * length(examples);
end
if ~isfield(options,'stepSize')
	options.stepSize = 1e-4;
end
if ~isfield(options,'verbose')
	options.verbose = 0;
end
if nargin < 5
	nParam = max(examples{1}.edgeMap(:));
	w = zeros(nParam,1);
end

% L2-regularized NLL objective
if length(C) == 1
	C = C * ones(size(w));
end
if usePL
	objFun = @(w,ex,t) penalizedL2(w,@UGM_CRFcell_PseudoNLL,C,{ex});
else
	objFun = @(w,ex,t) penalizedL2(w,@UGM_CRFcell_NLL,C,{ex},inferFunc);
end

% SGD
[w,fAvg] = sgd(examples,objFun,w,options);

% NLL of learned model
if usePL
	nll = UGM_CRFcell_PseudoNLL(w,examples);
else
	nll = UGM_CRFcell_NLL(w,examples,inferFunc);
end

		
