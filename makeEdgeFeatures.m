function Xedge = makeEdgeFeatures(Xnode,edgeEnds)
%
% Computes the edge features for the Weizmann dataset.
%
% Xnode : 1 x nFeat x nNode matrix of node features
% edgeEnds : nEdge x 2 matrix of edge ends

Xedge = [Xnode(1,:,edgeEnds(:,1)) Xnode(1,:,edgeEnds(:,2))];
