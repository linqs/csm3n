function [x, fAvg, fVec] = sgd(data, objFun, x0, options)
% 
% Performs stochastic gradient descent (SGD).
% 
% data : n x 1 cell array of data points
% objFun : objective function of the form:
%			 function [f, g] = objFun(x, ex, varargin), where
%			   x : current position
%			   ex : random example
%			   f : function value
%			   g : gradient w.r.t. x
% x0 : initial position
% options : optional arguments
% 			  maxIter : max iterations (def=n)
% 			  stepSize : step size (def=1)
% 			  verbose : display iteration info (def=0)
		  
%% Parse input

assert(nargin >= 3, 'USAGE: sgd(data,objFun,x0)');

n = length(data);

if nargin < 4
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
	[f, g] = objFun(x, ex); % TODO: support variable output functions
	
	% Update estimate of function value
	fAvg = (1/t) * f + ((t-1)/t) * fAvg;
	
	% Update point
	x = x - (options.stepSize / t) * g;
	
	if nargout >= 3
		fVec(t) = f;
	end
	
	if options.verbose
		fprintf('Iter = %d of %d (ex %d: f = %f, fAvg = %f)\n', t, options.maxIter, i, f, fAvg);
	end
	
end


