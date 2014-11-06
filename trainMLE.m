function [w,nll,iters,funcnt] = trainMLE(examples, inferFunc, C, options, w)
%
% Trains an MRF using MLE.
%
% examples : cell array of examples
% inferFunc : inference function (0: use pseudolikelihood)
% C : optional regularization constant or vector (def: 1)
% options : optimization options (optional)
% w : init weights (optional: def=zeros)

% parse input
assert(nargin >= 2,'USAGE: trainMLE_lbfgs(examples,inferFunc)')
usePL = ~isa(inferFunc,'function_handle');

% regularization param
if nargin < 3
	C = 1;
end

% optimization options
if nargin < 4 || ~isstruct(options)
	options = struct();
end
options.Method = 'lbfgs';
if ~isfield(options,'Display')
	options.Display = 'off';
end

% inital point
if nargin < 5
	nParam = max(examples{1}.edgeMap(:));
	w = zeros(nParam,1);
end

% L2 regularization
if length(C) == 1
	C = C * ones(size(w));
end
if usePL
	obj = @(w) penalizedL2(w,@UGM_CRFcell_PseudoNLL,C,examples);
else
	obj = @(w) penalizedL2(w,@UGM_CRFcell_NLL,C,examples,inferFunc);
end

% optimize
[w,~,~,output] = minFunc(obj,w,options);
iters = output.iterations;
funcnt = output.funcCount;

% compute NLL of solution
if usePL
	nll = UGM_CRFcell_PseudoNLL(w,examples);
else
	nll = UGM_CRFcell_NLL(w,examples,inferFunc);
end

