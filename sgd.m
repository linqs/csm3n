function [x, fAvg, fVec] = sgd(data, objFun, x0, projFun, options)
% 
% Performs stochastic (sub)gradient descent (SGD) with optional projection.
% 
% data : n x 1 cell array of data points
% objFun : objective function of the form:
%			 function [f, g] = objFun(x, ex, t), where
%			   x : current position
%			   ex : random example
%			   t : current iteration
%			   f : function value
%			   g : gradient w.r.t. x
% x0 : initial position
% projFun : optional projection function of the form:
%			 function w = projFun(v), where
%			   v : possibly infeasible point
%			   w : projection
%			If not used, pass 0 or empty array
% options : optional arguments
% 			  maxIter : max iterations (def=n)
% 			  stepSize : step size; can be vector or scalar (def=1)
% 			  verbose : display iteration info (def=0)
		  
%% Parse input

assert(nargin >= 3, 'USAGE: sgd(data,objFun,x0)');

n = length(data);

if ~exist('projFun','var') || ~isa(projFun,'function_handle')
	doProject = 0;
else
	doProject = 1;
end

if ~exist('options','var')
	options = struct();
end
if ~isfield(options,'maxIter')
	options.maxIter = n;
end
if ~isfield(options,'stepSize')
	options.stepSize = 1;
end
if ~isfield(options,'verbose')
	options.verbose = 0;
end


%% Main loop

fAvg = 0;
x = x0;
if nargout >= 3
	fVec = zeros(options.maxIter,1);
end

for t = 1:options.maxIter
	
	% Compute objective for random data point
	i = ceil(rand() * n);
	ex = data{i};
	[f, g] = objFun(x, ex, t); % TODO: support variable output functions
	
	% Update estimate of function value
	fAvg = (1/t) * f + ((t-1)/t) * fAvg;
	
	% Update point
	x = x - (options.stepSize ./ sqrt(t)) .* g;
	
	% Project into feasible region
	if doProject
		x = projFun(x);
	end
	
	if nargout >= 3
		fVec(t) = f;
	end
	
	if options.verbose
		fprintf('Iter = %d of %d (ex %d: f = %f, fAvg = %f)\n', t, options.maxIter, i, f, fAvg);
	end
	
end


