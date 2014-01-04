function inds = localIndex(i, s, nNode)

% Returns the index (or indices) of node i being assigned state s, where
% there are nNode nodes.

inds = (s - 1) .* nNode + i;
