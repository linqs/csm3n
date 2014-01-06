function [w, kappa, f] = trainVCTSM(examples, C, w, kappa)

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
% C : regularization constant or vector

% dimensions
nEx = length(examples);
nParam = max(examples{1}.edgeMap(:));
nCon = 0;
for i = 1:nEx
	nCon = nCon + length(examples{i}.beq);
end

% initial point
x0 = zeros(nParam+nCon+1,1);
if nargin >= 3
	x0(1:nParam) = w;
end
if nargin >= 4
	x0(nParam+1) = kappa;
else
	x0(nParam+1) = 1;
end

% optimization options
clear options;
options.Method = 'lbfgs';
options.Corr = 200;
options.LS_type = 0;
options.LS_interp = 0;
options.Display = 'off';
% options.maxIter = 8000;
% options.MaxFunEvals = 8000;
options.progTol = 1e-6;
options.optTol = 1e-3;

% run optimization
obj = @(y, varargin) vctsmObj(y, examples, C, varargin);
[x,f] = minFunc(obj, x0, options);

% parse optimization output
w = x(1:nParam);
kappa = exp(x(nParam+1));
% lambda = x(nParam+2:end);



