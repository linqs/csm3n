function [w, fAvg] = trainM3N(examples, decodeFunc, C, options, w)
%
% Trains an MRF using max-margin formulation.
%
% examples : nEx x 1 cell array of examples, each containing:
%	Fx : nParam x length(oc) feature map
%	suffStat : nParam x 1 vector of sufficient statistics (i.e., Fx * oc)
%	Ynode : nState x nNode overcomplete matrix representation of labels
% decodeFunc : decoder function
% C : optional regularization constant or vector (def: 1)
% options : optional struct of optimization options for subgradient descent:
% 			maxIter : iterations (def: 100*length(examples))
% 			stepSize : step size (def: 1)
% 			verbose : verbose mode (def: 0)
% w : init weights (optional: def=zeros)

% parse input
assert(nargin >= 2, 'USAGE: trainM3N(examples,decodeFunc)')
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
	options.stepSize = 1;
end
if ~isfield(options,'verbose')
	options.verbose = 0;
end
if nargin < 5
	nParam = max(examples{1}.edgeMap(:));
	w = zeros(nParam,1);
end

% Use projected subgradient descent for 1 training example;
% otherwise, use stochastic subgradient.
if length(examples) == 1
	objFun = @(x) m3nObj(x, examples, C, decodeFunc);
	[w,fAvg] = pgd(objFun, [], w, options);
else
	objFun = @(x, ex, t) m3nObj(x, {ex}, C, decodeFunc);
	[w,fAvg] = sgd(examples, objFun, w, [], options);
end

		
