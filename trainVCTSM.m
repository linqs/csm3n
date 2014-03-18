function [w, kappa, fAvg] = trainVCTSM(examples, inferFunc, C1, C2, options, w, kappa)

% Optimizes the VCTSM objective, learning the optimal (w,kappa).
%
% examples : nEx x 1 cell array of examples, each containing:
%	Fx : nParam x length(oc) feature map
%	suffStat : nParam x 1 vector of sufficient statistics (i.e., Fx * oc)
% inferFunc : inference function used for convexified inference
% C1 : optional regularization constant (def: 1),
%		controls tradeoff between regularizer and loss
% C2 : optional regularization constant (def: 1),
%		controls tradeoff between weight norm and convexity
% options : optional struct of optimization options for SGD:
% 			maxIter : iterations of SGD (def: 500*length(examples))
% 			stepSize : SGD step size (def: 1e-6)
% 			verbose : verbose mode (def: 0)
% w : init weights (optional: def=zeros)
% kappa : init kappa (optional: def=1)

% parse input
assert(nargin >= 2, 'USAGE: trainVCTSM(examples,inferFunc)')

nEx = length(examples);
nParam = max(examples{1}.edgeMap(:));
nCon = 0;
for i = 1:nEx
	nCon = nCon + length(examples{i}.beq);
end
if nargin < 3
	C1 = 1;
end
if nargin < 4
	C2 = 1;
end
if nargin < 5 || ~isstruct(options)
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
if nargin < 6 || isempty(w)
	w = zeros(nParam,1);
end
if nargin < 7 || isempty(kappa)
	kappa = 1;
end

% initial position
x0 = [w ; log(kappa)];

% SGD
objFun = @(x, ex, t) vctsmObj(x, {ex}, C1, C2, inferFunc);
[x,fAvg] = sgd(examples, objFun, x0, options);

% parse optimization output
w = x(1:nParam);
kappa = exp(x(nParam+1));



