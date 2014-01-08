function [x, fAvg] = sgd(data, objFun, x0, options)
% 
% Performs stochastic gradient descent (SGD).
% 
% data : n x 1 cell array of data points
% objFun : objective function of the form:
%			 function [f, g] = objFun(x, ex), where
%			   x : current position
% 			   ex : random example
% x0 : initial position
% options : optional arguments
% 			  maxIter : max iterations (def=n)
% 			  stepSize : step size (def=1e-4)
% 			  verbose : display iteration info (def=0)
		  
%% Parse input

assert(nargin >= 3, 'USAGE: sgd(data,obj,x0)');

n = length(data);

if nargin < 4
	options = {};
end

if isfield(options,'maxIter')
	maxIter = options.maxIter;
else
	maxIter = n;
end

if isfield(options,'stepSize')
	stepSize = options.stepSize;
else
	stepSize = 1e-4;
end

if isfield(options,'verbose')
	verbose = options.verbose;
else
	verbose = 0;
end


%% Main loop

fAvg = 0;
x = x0;

for t = 1:maxIter
	
	% Compute objective for random data point
	i = ceil(rand() * n);
	ex = data{i};
	[f, g] = objFun(x, ex); % TODO: support variable argument functions
	
	% Update estimate of function value and parameters
	fAvg = (1/t) * f + ((t-1)/t) * fAvg;
	x = x - (stepSize / t) * g;
	
	if verbose
		fprintf('Iter = %d of %d (ex %d: f = %f, fAvg = %f)\n', t, maxIter*n, i, f, fAvg);
	end
end


