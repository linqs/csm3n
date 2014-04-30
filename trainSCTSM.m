function [w, fAvg] = trainSCTSM(examples, inferFunc, kappa, C, options, w)

% Optimizes the VCTSM objective, learning the optimal (w,kappa).
%
% examples : nEx x 1 cell array of examples, each containing:
%	Fx : nParam x length(oc) feature map
%	suffStat : nParam x 1 vector of sufficient statistics (i.e., Fx * oc)
% inferFunc : inference function used for convexified inference
% kappa : modulus of convexity
% C : optional regularization constant or vector (def: 1)
% options : optional struct of optimization options for SGD:
% 			maxIter : iterations of SGD (def: 500*length(examples))
% 			stepSize : SGD step size (def: 1e-6)
% 			verbose : verbose mode (def: 0)
% w : init weights (optional: def=zeros)

% parse input
assert(nargin >= 3, 'USAGE: trainSCTSM(examples,inferFunc,kappa)')

nEx = length(examples);
nParam = max(examples{1}.edgeMap(:));
nCon = 0;
for i = 1:nEx
	nCon = nCon + length(examples{i}.beq);
end
if nargin < 4
	C = 1;
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

% Use projected subgradient descent for 1 training example;
% otherwise, use stochastic subgradient.
if length(examples) == 1
	objFun = @(x) sctsmObj(x, examples, C, inferFunc, kappa);
	[w,fAvg] = pgd(objFun, [], w, options);
else
	objFun = @(x, ex, t) sctsmObj(x, {ex}, C, inferFunc, kappa);
	[w,fAvg] = sgd(examples, objFun, w, [], options);
end



