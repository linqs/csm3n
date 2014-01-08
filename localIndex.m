function inds = localIndex(i, s, nState)

% Returns the index (or indices) of node i being assigned state(s) s, where
% there are nState states per node.

inds = (i - 1) * nState + s;
