function [w, fAvg] = trainCSCACC(examples_l, examples_u, decodeFunc, C_w, C_s, optSGD, optPGD, w)
%
% CACC training with stability regularization.
%
% examples_l : cell array of labeled examples
% examples_u : cell array of unlabeled examples
% decodeFunc : decoder function
% C_w : weight regularization constant or nParam x 1 vector (optional: def=nNode of first example)
% C_s : stability regularization constant (optional: def=0.1)
% optSGD : optional struct of optimization options for SGD:
% 			maxIter : iterations of SGD (def: 100*length(examples_l))
% 			stepSize : SGD step size (def: 1e-4)
% 			verbose : verbose mode (def: 0)
% optPGD : optional struct of optimization options for stabilityObj
% w : init weights (optional: def=zeros)

% parse input
assert(nargin >= 3, 'USAGE: trainM3N(examples_l,examples_u,decodeFunc)')
if nargin < 4
	C_w = examples_l{1}.nNode;
end
if nargin < 5
	C_s = 0.1;
end
if nargin < 6 || ~isstruct(optSGD)
	optSGD = struct();
end
if ~isfield(optSGD,'maxIter')
	optSGD.maxIter = 100 * length(examples_l);
end
if ~isfield(optSGD,'stepSize')
	optSGD.stepSize = 1e-4;
end
if ~isfield(optSGD,'verbose')
	optSGD.verbose = 0;
end
if nargin < 7
	optPGD = struct();
end
if nargin < 8
	nParam = max(examples_l{1}.edgeMap(:));
	w = zeros(nParam,1);
end

% SGD
objFun = @(x,ex,t) cscaccSGDObj(x,ex,examples_u,decodeFunc,C_w,C_s,optPGD);
[w,fAvg] = sgd(examples_l,objFun,w,optSGD);


% Subroutine for L2-regularized M3N objective
function [f, g] = cscaccSGDObj(w, ex_l, examples_u, decodeFunc, C_w, C_s, optPGD)
	
	% pick random unlabeled point
	i = ceil(rand() * length(examples_u));
	ex_u = examples_u{i};
	
	% compute CSM3N objective
	[f,g] = cscaccObj(w,{ex_l},{ex_u},decodeFunc,C_w,C_s,optPGD);

