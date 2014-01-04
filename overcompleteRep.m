function [ocrep] = overcompleteRep(Y, nState, vectorize)

% Converts a vector of values to overcomplete representation.
%
% vals : nVars x 1 vector of values
% nLabel : number of labels per variable (must be uniform)
% vectorize : convert output to vector (optional: def=1)
%
% ocrep : if vector: nState*nNode x 1 overcomplete representation
%		  if matrix: nState x nNode overcomplete representation

if nargin < 3
	vectorize = 1;
end

nNode = length(Y);

ocrep = zeros(nState,nNode);

for i = 1:nNode
	ocrep(Y(i),i) = 1;
end

if vectorize
	ocrep = ocrep(:);
end
