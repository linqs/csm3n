function [w, kappa, f] = trainVCTSM_lbfgs(examples, inferFunc, C1, C2, options, w, kappa)

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
% options : optional struct of optimization options for LBFGS:
% w : init weights (optional: def=zeros)
% kappa : init kappa (optional: def=1)

assert(nargin >= 2,'USAGE: trainVCTSM_lbfgs(examples,inferFunc)')

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
% 	options.progTol = 1e-10;
% end
% if ~isfield(options,'optTol')
% 	options.optTol = 1e-10;
% end
if nargin < 6 || isempty(w)
	w = zeros(nParam,1);
end
if nargin < 7 || isempty(kappa)
	kappa = 1;
end

% initial position
x0 = [w ; log(kappa)];

% run optimization
objFun = @(x, varargin) vctsmObj(x, examples, C1, C2, inferFunc, varargin{:});
[x,f] = minFunc(objFun, x0, options);

% parse optimization output
w = x(1:nParam);
kappa = exp(x(nParam+1));



