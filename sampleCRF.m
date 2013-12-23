% Generates samples of a CRF with E-R graph structure.

clear

% structural constants
nNode = 100;
sparsity = 0.1;
nFeat = 1; % # local features
nStateX = 2^nFeat;
nStateY = 2;
nState = max(nStateX,nStateY);

% sampling params
nSamp = 100;
burnin = 10;

%% STRUCTURE

% make a dumb E-R graph
G = sprand(nNode,nNode,sparsity);
G = triu(G,1) + triu(G,1)';

% adjacency graph of CRF
adj = sparse(2*nNode,2*nNode);
adj(1:nNode,1:nNode) = G;
adj(1:nNode,nNode+1:end) = speye(nNode);
adj(nNode+1:end,1:nNode) = speye(nNode);

% get edge structure
edgeStruct = UGM_makeEdgeStruct(adj,[repmat(nStateY,nNode,1); repmat(nStateX,nNode,1)],0,nSamp);
nEdge = edgeStruct.nEdges;

%% MODEL

% node potentials don't matter
nodePot = ones(2*nNode,nState);

% edge potentials
wloc = eye(nStateY,nStateX) + 0.5*randn(nStateY,nStateX);
wrel = eye(nStateY) + 0.5*randn(nStateY,nStateY);
edgePot = zeros(nStateY,nStateX,nEdge);
for e=1:nEdge
    ends = edgeStruct.edgeEnds(e,:);
	n2 = edgeStruct.edgeEnds(e,2);
    if n2 > nNode
        edgePot(1:nStateY,1:nStateX,e) = exp(wloc);
    else
        edgePot(1:nStateY,1:nStateY,e) = exp(wrel);
    end
end

%% SAMPLING

% Gibbs sampling
samples = UGM_Sample_Gibbs(nodePot,edgePot,edgeStruct,burnin);

Y = int32(samples(1:nNode,:));
X = zeros(nSamp,nStateX,nNode);
for i = 1:nSamp
	X(i,:,:) = overcompleteRep(samples(nNode+1:end,i),nStateX,0);
end
X = int32(X);

save('crf_samples.mat','G','Y','X','nStateX','nStateY','wloc','wrel','nodePot','edgePot');

