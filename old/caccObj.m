function [f, g] = caccObj(w, examples, decodeFunc, C, options, varargin)
% 
% Computes the CACC objective and gradient.
% 
% w : nParam x 1 vector of weights
% examples : cell array of labeled examples
% decodeFunc : decoder function
% C : weight regularization constant or nParam x 1 vector
% options : optional struct of optimization options for stabilityObj

if nargin < 5
	options = struct();
end

% L2 weight regularization
f = 0.5 * (C .* w)' * w;
g = C .* w;

% stability objective
nEx = length(examples);
for i = 1:nEx
	ex = examples{i};
	[l,sg] = stabilityObj(w,ex,ex.Y,decodeFunc,options,varargin{:});
	f = f + l / nEx;
	g = g + sg / nEx;
end


	