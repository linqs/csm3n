% This script simply tests the functionality of Mark Schmidt's toolbox

clear

% structural constants
nNodes = 100;
sparsity = 0.1;
nFeat = 20; % each feature is assumed Boolean

% make a dumb E-R graph
G = rand(nNodes) <= sparsity;
G = triu(G, 1) + triu(G, 1)';

% get edge structure
edgeStruct = UGM_makeEdgeStruct(G,2^(nFeat+1));