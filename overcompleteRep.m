function [ocrep] = overcompleteRep(y, nState, vectorize)

% Converts a vector of values to overcomplete representation.
%
% vals : nNode x 1 vector of state values
% nState : number of states per variable (must be uniform)
% vectorize : convert output to vector (optional: def=1)
%
% ocrep : if vector: nState*nNode x 1 overcomplete representation
%		  if matrix: nState x nNode overcomplete representation

if nargin < 3
	vectorize = 1;
end

nNode = length(y);

ocrep = zeros(nState,nNode);

for n = 1:nNode
	ocrep(y(n),n) = 1;
end

if vectorize
	ocrep = ocrep(:);
end
