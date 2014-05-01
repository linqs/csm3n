function [x, f, fvec] = pgd(objFun, projFun, x0, options)
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
%			  plotObj : plot objective (number indicates which plot to use)
		  
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
if ~isfield(options,'plotObj')
	options.plotObj = 0;
end


%% Main loop

% Initial point
x = x0;
[f, g] = objFun(x);
if options.verbose
	fprintf('Initial point: f = %f\n', f);
end
if nargout >= 3 || options.plotObj ~= 0
	fvec = zeros(options.maxIter+1,1);
	fvec(1) = f;
end

bestFval = f;
bestXval = x;

% normW = norm(x(1:end-1));
% kappa = exp(x(end));
% stability = (normW / kappa)^2;

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
	if nargout >= 3 || options.plotObj ~= 0
		fvec(t+1) = f;
% 		normW(end+1) = norm(x(1:end-1));
% 		kappa(end+1) = exp(x(end));
% 		stability(end+1) = (normW / kappa)^2;
	end
	
	% Keep track of best
	if bestFval > f
		bestFval = f;
		bestXval = x;
	end
	
	if options.verbose
		fprintf('Iter = %d of %d: f = %f\n', t, options.maxIter, f);
	end
	
	if options.plotObj ~= 0 && mod(t,100) == 0
		figure(options.plotObj);
		plot(1:t+1,fvec(1:t+1));
% 		plotyy(1:t+1,fvec(1:t+1),1:t+1,stability(1:t+1));
% 		figure(options.plotObj+1);
% 		plotyy(1:t+1,normW,1:t+1,kappa);
		drawnow;
	end

end

if options.plotObj ~= 0
	figure(options.plotObj);
	plot(1:length(fvec),fvec);
% 	plotyy(1:length(fvec),fvec,1:length(stability),stability);
% 	figure(options.plotObj+1);
% 	plotyy(1:t+1,normW,1:t+1,kappa);
	drawnow;
end

% Return best
if options.returnBest
	f = bestFval;
	x = bestXval;
end


