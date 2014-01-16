function [x, f] = pgd(objFun, projFun, x0, options)
% 
% Performs projected gradient descent (PGD).
% 
% objFun : objective function of the form:
%			 function [f, g] = objFun(x, varargin), where
%			   x : current position
%			   f : function value
%			   g : gradient w.r.t. x
% projFun : projection function of the form:
%			 function w = projFun(v, varargin), where
%			   v : possibly infeasible point
%			   w : projection
% x0 : initial position
% options : optional arguments
% 			  maxIter : max iterations (def=100)
% 			  stepSize : step size (def=1)
% 			  fTol : function tolerance (def=1e-4)
% 			  verbose : display iteration info (def=0)
		  
%% Parse input

assert(nargin >= 3, 'USAGE: pgd(objFun,projFun,x0)');

if nargin < 4
	options = {};
end

if isfield(options,'maxIter')
	maxIter = options.maxIter;
else
	maxIter = 100;
end

if isfield(options,'stepSize')
	stepSize = options.stepSize;
else
	stepSize = 1;
end

if isfield(options,'fTol')
	fTol = options.fTol;
else
	fTol = 1e-4;
end

if isfield(options,'verbose')
	verbose = options.verbose;
else
	verbose = 0;
end


%% Main loop

% Initial point
x = x0;
[f, g] = objFun(x);
if verbose
	fprintf('Initial point: f = %f\n', f);
end

% Iterative updates
for t = 1:maxIter
	
	% Update point
	x = x - (stepSize / t) * g;
	
	% Project new point into feasible region
	x = projFun(x);
	
	% Compute objective
	[f, g] = objFun(x);

	if verbose
		fprintf('Iter = %d of %d: f = %f\n', t, maxIter, f);
	end
		
end


