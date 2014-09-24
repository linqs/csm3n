function Xedge = makeEdgeFeatures(Xnode,edgeEnds,addBias)
%
% Computes the edge features for the Weizmann dataset.
%
% Xnode : 1 x nFeat x nNode matrix of node features
% edgeEnds : nEdge x 2 matrix of edge ends
% addBias : (optional) adds a bias feature (def: 0)

if ~exist('addBias','var')
	addBias = 0;
end

assert(~isempty(Xnode) || addBias, 'Cannot create empty edge features');

if isempty(Xnode)
	Xedge = ones(1,1,size(edgeEnds,1));
else
	Xedge = [ones(1,addBias,size(edgeEnds,1)) Xnode(1,:,edgeEnds(:,1)) Xnode(1,:,edgeEnds(:,2))];
end
