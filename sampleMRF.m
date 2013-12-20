% Generates samples of a CRF with E-R graph structure.

clear

% structural constants
nNode = 100;
sparsity = 0.1;
nFeat = 1; % # local features
nState = 2;

% sampling params
nSamp = 100;
burnin = 10;

%% STRUCTURE

% make a dumb E-R graph
G = sprand(nNode,nNode,sparsity);
G = triu(G,1) + triu(G,1)';

% adjacency graph
adj = sparse(nNode*(1+nFeat),nNode*(1+nFeat));
adj(1:nNode,1:nNode) = G;
for i = 1:nNode
	adj(i,nNode+(i-1)*nFeat+1:nNode+i*nFeat) = 1;
	adj(nNode+(i-1)*nFeat+1:nNode+i*nFeat,i) = 1;
end

% get edge structure
edgeStruct = UGM_makeEdgeStruct(adj,nState);
nEdge = edgeStruct.nEdges;


%% MODEL

% node potentials don't matter
nodePot = ones(nNode*(1+nFeat),nState);

% edge potentials
wloc = [1 .1; .1 1];
wrel = [.6 .4; .4 .6];
% wloc = eye(nState) + 0.1*randn(nState);
% wrel = eye(nState) + 0.1*randn(nState);
edgePot = ones(nState,nState,nEdge);
edgeType = zeros(nEdge,1);
for e = 1:nEdge
	n2 = edgeStruct.edgeEnds(e,2);
	if n2 > nNode
		edgePot(:,:,e) = exp(wloc);
		edgeType(e) = mod(n2-nNode-1, nFeat) + 2;
	else
		edgePot(:,:,e) = exp(wrel);
		edgeType(e) = 1;
	end
end


%% SAMPLING

% Gibbs sampling
edgeStruct.useMex = 1; % mex version difficult to debug
edgeStruct.maxIter = nSamp;
samples = UGM_Sample_Gibbs(nodePot,edgePot,edgeStruct,burnin);

Y = samples(1:nNode,:);
X = samples(nNode+1:end,:);

save mrf_samples.mat

