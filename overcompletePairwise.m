function [ocrep] = overcompletePairwise(Y, edgeStruct)

% Converts a vector of values to overcomplete pairwise representation.
%
% vals : nVars x 1 vector of values
% edgeStruct : edge structure
%
% ocrep : (nNode*nState + nEdge*nState^2) x 1 overcomplete representation

nNode = double(edgeStruct.nNodes);
nEdge = edgeStruct.nEdges;
nState = double(max(edgeStruct.nStates));
assert(nState == min(edgeStruct.nStates), 'overcompletePairwise assumes uniform domain for Y')

ocrep = zeros(nNode*nState + nEdge*nState^2, 1);

for i = 1:nNode
	ocrep(localIndex(i,Y(i),nNode)) = 1;
end

for e = 1:nEdge
	i = edgeStruct.edgeEnds(e,1);
	j = edgeStruct.edgeEnds(e,2);
	ocrep(pairwiseIndex(e,Y(i),Y(j),nNode,nState)) = 1;
end

