function Xedge = makeRbfEdgeFeatures(edgeEnds,pixelInt,edgeDirections,scale)
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

sim = exp(- scale * (pixelInt(edgeEnds(:,1)) - pixelInt(edgeEnds(:,2))).^2);

Xedge = zeros(1, 4, size(edgeEnds,1));

lookupMask = sparse(double(edgeEnds(:,1)), double(edgeEnds(:,2)), ...
    true(size(edgeEnds,1),1), size(edgeDirections, 1), size(edgeDirections, 2));

direction = full(edgeDirections(lookupMask));


% populate features for each direction
for i = 1:4
    idx = direction==i;
    Xedge(1, i, idx) = sim(idx);
end

