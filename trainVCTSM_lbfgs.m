function [w, kappa, f] = trainVCTSM_lbfgs(examples, inferFunc, C1, C2, options, w, kappa)

% Optimizes the VCTSM objective with LBFGS, learning the optimal (w,kappa).
%
% examples : nEx x 1 cell array of examples, each containing:
%	Fx : nParam x length(oc) feature map
%	suffStat : nParam x 1 vector of sufficient statistics (i.e., Fx * oc)
%	Ynode : nState x nNode overcomplete matrix representation of labels
% inferFunc : inference function used for convexified inference
% C1 : optional regularization constant (def: 1),
%		controls tradeoff between regularizer and loss
% C2 : optional regularization constant (def: 1),
%		controls tradeoff between weight norm and convexity
% options : optional struct of optimization options for LBFGS
%			Important options (for complete list, see minFunc)
%				Display : display mode
%				MaxIter : maximum iterations
%				MaxFunEvals : maximum function evals
% w : init weights (optional: def=zeros)
% kappa : init kappa (optional: def=1)

assert(nargin >= 2,'USAGE: trainVCTSM_lbfgs(examples,inferFunc)')

nParam = max(examples{1}.edgeMap(:));

if ~exist('C1','var') || isempty(C1)
	C1 = 1;
end
if ~exist('C2','var') || isempty(C2)
	C2 = 1;
end

if ~exist('options','var') || ~isstruct(options)
	options = struct();
end
options.Method = 'lbfgs';
% options.LS_type = 0;
% options.LS_interp = 0;
if ~isfield(options,'verbose')
	options.Display = 0;
end

if ~exist('w','var') || isempty(w)
	w = zeros(nParam,1);
end

if ~exist('kappa','var') || isempty(kappa)
	kappa = 1;
end

% % Unconstrained optimization (in log space)
% objFun = @(x, varargin) vctsmObj_log(x, examples, C1, C2, inferFunc, varargin{:});
% x0 = [w ; log(kappa)];
% [x,f] = minFunc(objFun, x0, options);
% w = x(1:end-1);
% kappa = exp(x(end));

% Constrained optimization
objFun = @(x, varargin) vctsmObj(x, examples, C1, C2, inferFunc, varargin{:});
x0 = [w ; kappa];
lb = -inf(size(x0)); lb(end) = 1e-10;
ub = inf(size(x0));
[x,f] = minConf_TMP(objFun, x0, lb, ub, options);
w = x(1:end-1);
kappa = x(end);



