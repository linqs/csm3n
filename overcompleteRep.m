function [ocrep] = overcompleteRep(vals, nLabel)

% Converts a vector of values to overcomplete representation.
%
% vals : nVars x 1 vector of values
% nLabel : number of labels per variable (must be uniform)
%
% ocrep : n*nLabels x 1 overcomplete representation

nVars = length(vals);

ocrep = zeros(nVars*nLabel,1);

for i = 1:nVars
	ocrep((i-1)*nLabel+vals(i)) = 1;
end

