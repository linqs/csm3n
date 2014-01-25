function Xedge = makeEdgeFeatures(Xnode,edgeEnds,pixelInt,scale)
%
% Computes the edge features for the Weizmann dataset.
%
% Xnode : 1 x nFeat x nNode matrix of node features
% edgeEnds : nEdge x 2 matrix of edge ends
% pixelInt : nNode x 1 vector of pixel intensities
% scale : (optional) scales RBF similarity (def: 1)
% 			rbf = exp( scale * (x_i - x_j)^2 )

if nargin < 4
	scale = 1;
end

sim = exp(scale * (pixelInt(edgeEnds(:,1)) - pixelInt(edgeEnds(:,2))).^2);
Xedge = [Xnode(1,:,edgeEnds(:,1)) Xnode(1,:,edgeEnds(:,2)) reshape(sim,1,1,[])];

