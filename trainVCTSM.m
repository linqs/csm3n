function [w, kappa, f] = trainVCTSM(examples, inferFunc, C, options, w, kappa)

% Optimizes the VCTSM objective with LBFGS, learning the optimal (w,kappa).
%
% examples : nEx x 1 cell array of examples, each containing:
%	Fx : nParam x length(oc) feature map
%	suffStat : nParam x 1 vector of sufficient statistics (i.e., Fx * oc)
%	Ynode : nState x nNode overcomplete matrix representation of labels
% inferFunc : inference function used for convexified inference
% C : optional regularization constant (def: 1)
% options : optional struct of optimization options for LBFGS
%			Important options (for complete list, see minFunc)
%				Display : display mode
%				MaxIter : maximum iterations
%				MaxFunEvals : maximum function evals
% w : init weights (optional: def=zeros)
% kappa : init kappa (optional: def=1)

assert(nargin >= 2,'USAGE: trainVCTSM(examples,inferFunc)')

nParam = max(examples{1}.edgeMap(:));

if ~exist('C','var') || isempty(C)
	C = 1;
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
if isfield(options,'plotObj') && options.plotObj ~= 0
	if ~isfield(options,'plotRefresh')
		options.plotRefresh = 100;
	end
	options.traceFunc = @(trace) plotFunc(trace,options.plotRefresh,options.plotObj);
end

if ~exist('w','var') || isempty(w)
	w = zeros(nParam,1);
end

if ~exist('kappa','var') || isempty(kappa)
	kappa = 1;
end

% Unconstrained optimization (in log space)
objFun = @(x, varargin) vctsmObj_log(x, examples, C, inferFunc, varargin{:});
x0 = [w ; log(kappa)];
[x,f] = minFunc(objFun, x0, options);
w = x(1:end-1);
kappa = exp(x(end));

% % Constrained optimization
% objFun = @(x, varargin) vctsmObj(x, examples, C1, inferFunc, varargin{:});
% x0 = [w ; kappa];
% % lb = -inf(size(x0)); lb(end) = 1e-10;
% % ub = inf(size(x0));
% % [x,f] = minConf_TMP(objFun, x0, lb, ub, options);
% [x,f] = minConf_PQN(objFun, x0, @projFun, options);
% w = x(1:end-1);
% kappa = x(end);


%% Projection function (ensures that kappa is positive)
function x = projFun(x)

if x(end) < 1e-10
	x(end) = 1e-10;
end

%% Plotting function
function plotFunc(trace,plotRefresh,fig)

t = length(trace.fval);
if mod(t,plotRefresh) == 0
	figure(fig);
	hAx = plotyy(1:t,trace.fval, 1:t,trace.normx);
	ylabel(hAx(1),'Objective'); ylabel(hAx(2),'norm(x)');
	title('VCTSM Objective')
	drawnow;
end



