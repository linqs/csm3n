function [f, g] = Xobj(Wnode, Wedge, Xnode, Ynode, Yedge, edgeEnds)
%
% Computes w'*f(x,y) and the gradient w.r.t. x.
%
% INPUT
% Wnode : nFeat x nState, node weights
% Wedge : 2*nFeat x nState^2, edge weights
% Xnode : nFeat x nNode, X node values
% Ynode : nState x nNode, Y node values
% Yedge : nState x nState x nEdge, Y edge values
% edgeEnds : nEdge x 2, edge ends
%
% OUTPUT
% g : nFeat x nNode gradient


nNode = size(Ynode,2);
nEdge = size(Yedge,3);

f = 0;
g = zeros(size(Xnode));

for i = 1:nNode
	f = f + sum(sum(Wnode .* (Xnode(:,i) * Ynode(:,i)')));
	g(:,i) = Wnode * Ynode(:,i);
end

for e = 1:nEdge
	i = edgeEnds(e,1);
	j = edgeEnds(e,2);
	y_e = reshape(Yedge(:,:,e),1,[]);
	f = f + sum(sum(Wedge(:,:,1) .* (Xnode(:,i) * y_e'))) ...
		  + sum(sum(Wedge(:,:,2) .* (Xnode(:,j) * y_e')));
	g(:,i) = g(:,i) + Wedge(:,:,1) * y_e;
	g(:,j) = g(:,j) + Wedge(:,:,2) * y_e;
end

