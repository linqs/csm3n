function y = decodeMarginals(mu, nNode, nState)
%
% Decodes a vector of marginals using the most likely state for each node.
%
% mu : vector of marginals
% nNode : number of nodes (i.e., local variables)
% nState : number of states per node

y = zeros(nNode,1);

for i = 1:nNode
	[~,y(i)] = max(mu(localIndex(i,1:nState,nState)));
end
