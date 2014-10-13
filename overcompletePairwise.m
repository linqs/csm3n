function [ocrep,nodeBel,edgeBel] = overcompletePairwise(y, nState, edgeStruct)

% Converts a vector of values to overcomplete pairwise representation.
%
% y : nNode x 1 vector of state values
% nState : number of states
% edgeStruct : edge structure
%
% ocrep : (nState*nNode + nState^2*nEdge) x 1 overcomplete representation
%  Equivalent to [reshape(nodeBel',[],1) ; edgeBel(:)] if beliefs are integral

nNode = edgeStruct.nNodes;
nEdge = edgeStruct.nEdges;
% assert(all(edgeStruct.nStates==nState), 'overcompletePairwise assumes uniform domain for Y')

nodeBel = zeros(nState,nNode);
edgeBel = zeros(nState,nState,nEdge);
for n = 1:nNode
	nodeBel(y(n),n) = 1;
end
for e = 1:nEdge
	n1 = edgeStruct.edgeEnds(e,1);
	n2 = edgeStruct.edgeEnds(e,2);
	edgeBel(y(n1),y(n2),e) = 1;
end
ocrep = [reshape(nodeBel,[],1) ; edgeBel(:)];

