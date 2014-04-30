function [x, f, fVec] = pgd(objFun, projFun, x0, options)
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
% 			  verbose : display iteration info (def=0)
%			  returnBest : return the best solution (def=0, returns last solution)
		  
%% Parse input

assert(nargin >= 3, 'USAGE: pgd(objFun,projFun,x0)');

if ~isa(projFun,'function_handle')
	doProject = 0;
else
	doProject = 1;
end

if ~exist('options','var')
	options = struct();
end
if ~isfield(options,'maxIter')
	options.maxIter = 100;
end
if ~isfield(options,'stepSize')
	options.stepSize = 1;
end
if ~isfield(options,'verbose')
	options.verbose = 0;
end
if ~isfield(options,'returnBest')
	options.returnBest = 0;
end

%% Main loop

% Initial point
x = x0;
[f, g] = objFun(x);
if options.verbose
	fprintf('Initial point: f = %f\n', f);
end
if nargout >= 3
	fVec = zeros(options.maxIter+1,1);
	fVec(1) = f;
end

bestFval = f;
bestXval = x;

% Iterative updates
for t = 1:options.maxIter
	
	% Update point
	x = x - (options.stepSize ./ t) .* g;
% 	x = x - (options.stepSize ./ sqrt(t)) .* g;
	
	% Project new point into feasible region
	if doProject
		x = projFun(x);
	end
	
	% Compute objective
	[f, g] = objFun(x);
	if nargout >= 3
		fVec(t+1) = f;
	end
	
	% Keep track of best
	if bestFval > f
		bestFval = f;
		bestXval = x;
	end
	
	if options.verbose
		fprintf('Iter = %d of %d: f = %f\n', t, options.maxIter, f);
	end

end

% Return best
if options.returnBest
	f = bestFval;
	x = bestXval;
end


