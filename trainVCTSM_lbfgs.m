function [w, kappa, f] = trainVCTSM_lbfgs(examples, C, options, w, kappa)

% Optimizes the VCTSM objective, learning the optimal (w,kappa).
%
% examples : nEx x 1 cell array of examples, each containing:
%	oc : full overcomplete vector representation of Y
%		 (including high-order terms)
%	ocLocalScope : number of local terms in oc
%	Aeq : nCon x length(oc) constraint A matrix
%	beq : nCon x 1 constraint b vector
%	F : nParam x length(oc) feature map
%	suffStat : nParam x 1 vector of sufficient statistics (i.e., F * oc)
% C : optional regularization constant or nParam x 1 vector (def: 1)
% options : optional optimization options
% w : optional init weight vector (def: 0)
% kappa : optional init convexity modulus (def: 1)

assert(nargin >= 2,'USAGE: trainVCTSM_lbfgs(examples)')

% dimensions
nEx = length(examples);
nParam = max(examples{1}.edgeMap(:));
nCon = 0;
for i = 1:nEx
	nCon = nCon + length(examples{i}.beq);
end

% regularization param
if nargin < 2
	C = 1;
end

% optimization options
if nargin < 3 || ~isstruct(options)
	options = struct();
end
options.Method = 'lbfgs';
options.LS_type = 0;
options.LS_interp = 0;
if ~isfield(options,'Display')
	options.Display = 'off';
end
% if ~isfield(options,'MaxIter')
% 	options.MaxIter = 1000;
% end
% if ~isfield(options,'MaxFunEvals')
% 	options.MaxFunEvals = 2000;
% end
% if ~isfield(options,'progTol')
% 	options.progTol = 1e-6;
% end
% if ~isfield(options,'optTol')
% 	options.optTol = 1e-3;
% end

% initial point
x0 = zeros(nParam+nCon+1,1);
if nargin >= 4
	x0(1:nParam) = w;
end
if nargin >= 5
	x0(nParam+1) = kappa;
else
	x0(nParam+1) = 1;
end

% run optimization
obj = @(y, varargin) vctsmObj(y, examples, C, varargin);
[x,f] = minFunc(obj, x0, options);

% parse optimization output
w = x(1:nParam);
kappa = exp(x(nParam+1));
% lambda = x(nParam+2:end);



