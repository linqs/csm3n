function [w, nll] = trainMLE_lbfgs(examples, inferFunc, C, w)
%
% Trains an MRF using MLE.
%
% examples : cell array of examples
% inferFunc : inference function (0: use pseudolikelihood)
% C : regularization constant or nParam x 1 vector (optional: def=nNode of first example)
% w : init weights (optional: def=zeros)

% parse input
assert(nargin >= 2,'USAGE: trainMLE(examples,inferFunc,C,w)')
usePL = ~isa(inferFunc,'function_handle');
if nargin < 3
	C = examples{1}.nNode;
end
if nargin < 4
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

% optimization options
clear options;
options.Display = 'off';

% Optimize
w = minFunc(obj,w,options);

if usePL
	nll = UGM_CRFcell_PseudoNLL(w,examples);
else
	nll = UGM_CRFcell_NLL(w,examples,inferFunc);
end

