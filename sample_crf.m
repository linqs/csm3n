% This script simply tests the functionality of Mark Schmidt's toolbox

clear

% structural constants
nNode = 100;
sparsity = 0.1;
nFeat = 1; % # local features
nStateX = 2^nFeat;
nStateY = 2;

% make a dumb E-R graph
G = sprand(nNode,nNode,sparsity);
G = triu(G,1) + triu(G,1)';

% adjacency graph of CRF
adj = sparse(2*nNode,2*nNode);
adj(1:nNode,1:nNode) = G;
adj(1:nNode,nNode+1:end) = speye(nNode);
adj(nNode+1:end,1:nNode) = speye(nNode);

% get edge structure
edgeStruct = UGM_makeEdgeStruct(adj,nStateX);
nEdge = edgeStruct.nEdges;

%% MODEL

% node potentials
% wpriorX = [.5 .5];
% wpriorY = [.5 .5];
% nodePotX = exp(repmat(wpriorX,nNode,1) + 0.05*randn(nNode,nStateX));
% nodePotY = exp(repmat(wpriorY,nNode,1) + 0.05*randn(nNode,nStateY));
% nodePot = [nodePotY; nodePotX];

% node potentials don't matter
nodePot = ones(2*nNode,nStateX);

% edge potentials
wloc = eye(nStateX,nStateY) + 0.5*randn(nStateX,nStateY);
wrel = eye(nStateY) + 0.5*randn(nStateY,nStateY);
edgePot = zeros(nStateX,nStateY,nEdge);
for e=1:nEdge
    ends = edgeStruct.edgeEnds(e,:);
    if (ends(1) > nNode) || (ends(2) > nNode)
        edgePot(:,:,e) = exp(wloc + 0.05*randn(nStateX,nStateY));
    else
        edgePot(:,:,e) = exp(wrel + 0.05*randn(nStateY,nStateY));
    end
end


%% SAMPLING

% Gibbs sampling
edgeStruct.maxIter = 100; % # samples
samples = UGM_Sample_Gibbs(nodePot,edgePot,edgeStruct,10);



