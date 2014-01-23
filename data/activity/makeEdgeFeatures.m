function Xedge = makeEdgeFeatures(Xnode,nodeId,edgeEnds)
%
% Computes the edge features for the Activity Detection dataset.
%
% Xnode : 1 x nFeat x nNode matrix of node features
% nodeId : nNode x 1 vector of node IDs
% edgeEnds : nEdge x 2 matrix of edge ends

nEdge = size(edgeEnds,1);
nFeat = size(Xnode,2);
Xedge = zeros(1,2*nFeat+1,nEdge);
for e = 1:nEdge
	n1 = edgeEnds(e,1);
	n2 = edgeEnds(e,2);
	sameId = nodeId(n1) == nodeId(n2);
	Xedge(1,:,e) = [Xnode(1,:,n1) Xnode(1,:,n2) sameId];
end
