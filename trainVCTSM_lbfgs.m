function [w, kappa, f] = trainVCTSM_lbfgs(examples, inferFunc, C, options, w, kappa)

% Optimizes the VCTSM objective, learning the optimal (w,kappa).
%
% examples : nEx x 1 cell array of examples, each containing:
%	F : nParam x length(oc) feature map
%	suffStat : nParam x 1 vector of sufficient statistics (i.e., F * oc)
% inferFunc : inference function used for convexified inference
% C : optional regularization constant or nParam x 1 vector (def: 1)
% options : optional optimization options
% w : optional init weight vector (def: 0)
% kappa : optional init convexity modulus (def: 1)

assert(nargin >= 2,'USAGE: trainVCTSM_lbfgs(examples,inferFunc)')

% dimensions
nEx = length(examples);
nParam = max(examples{1}.edgeMap(:));
nCon = 0;
for i = 1:nEx
	nCon = nCon + length(examples{i}.beq);
end

% regularization param
if nargin < 3
	C = 1;
end

% optimization options
if nargin < 4 || ~isstruct(options)
	options = struct();
end
options.Method = 'lbfgs';
options.LS_type = 0;
options.LS_interp = 0;
if ~isfield(options,'Display')
	options.Display = 'on';
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
x0 = zeros(nParam+1,1);
if nargin >= 5 && ~isempty(w)
	x0(1:nParam) = w;
end
if nargin >= 6 && ~isempty(kappa)
	x0(nParam+1) = log(kappa);
else
	x0(nParam+1) = 0;
end

% run optimization
obj = @(y, varargin) vctsmObj(y, examples, C, inferFunc, varargin{:});
[x,f] = minFunc(obj, x0, options);

% parse optimization output
w = x(1:nParam);
kappa = exp(x(nParam+1));



