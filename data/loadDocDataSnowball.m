function [examples, foldIdx] = loadDocDataSnowball(fName, nNet, nPC, jumpRate, seeds, makeEdgeDist, makeCounts, plotNets)

if ~exist('jumpRate','var') || isempty(jumpRate)
	jumpRate = 0;
end
if ~exist('seeds','var')
	seeds = [];
end
if ~exist('makeEdgeDist','var') || isempty(makeEdgeDist)
	makeEdgeDist = 0;
end
if ~exist('makeCounts','var') || isempty(makeCounts)
	makeCounts = 0;
end
if ~exist('plotNets','var') || isempty(plotNets)
	plotNets = 0;
end

% Load data
%	Assumes data contains 3 variables:
%		G : graph (nNode x nNode)
%		X : node features (nNode x nFeat)
%		y : node labels (nNode x 1)
load(fName);

% Constants
nState = max(unique(y));

% Remove diagonal from G
G = G - diag(diag(G));

% Snowball sample nNet networks
subgraphs = snowballSample(G,nNet,jumpRate,seeds,plotNets);

% Perform PCA on all observed features
Xpca = bsxfun(@minus,full(X),mean(X,1));
[V,~] = eigs(Xpca' * Xpca / (size(X,1)-1), nPC);
Xpca = Xpca * V;

% Create nNet network examples
examples = cell(nNet,1);
for i = 1:nNet
	
	edgeStruct = UGM_makeEdgeStruct(subgraphs(i).A,nState,1);
% 	[Aeq,beq] = pairwiseConstraints(edgeStruct);
	Aeq = []; beq = [];

	if makeEdgeDist
		edgeStruct.edgeDist = UGM_makeEdgeDistribution(edgeStruct);
	end
	
	if makeCounts
		[edgeStruct.nodeCount,edgeStruct.edgeCount] = UGM_ConvexBetheCounts(edgeStruct,1,.01,1);
	end
	
	% Node features are [bias pc_1 ... pc_k]
	Xnode = zeros(1,1+nPC,length(subgraphs(i).nodes));
	Xnode(1,:,:) = [ones(size(Xnode,3),1) Xpca(subgraphs(i).nodes,:)]';
	
	% Edge features are [bias cos_sim]
	X_i = X(subgraphs(i).nodes,:);
	Xmag = diag(sqrt(sum(X_i.^2,2)));
	Xsim = Xmag^-1 * (X_i * X_i') * Xmag^-1;
	edgeIdx = sub2ind(size(Xsim),edgeStruct.edgeEnds(:,1),edgeStruct.edgeEnds(:,2));
	Xedge = zeros(1,2,edgeStruct.nEdges);
	Xedge(1,:,:) = [ones(edgeStruct.nEdges,1) Xsim(edgeIdx)]';

	examples{i} = makeExample(Xnode,Xedge,y(subgraphs(i).nodes),nState,edgeStruct,Aeq,beq);

end

if nargout == 2
	% Generate nNet! folds of all permutations of 1:nNet, with
	%  1 train
	%  1 cv
	%  rest are test
	fold = 0;
	nets = 1:nNet;
	for tr = 1:nNet
		for cv = 1:nNet
			if tr == cv
				continue
			end
			fold = fold + 1;
			foldIdx(fold).tridx = tr;
			foldIdx(fold).ulidx = [];
			foldIdx(fold).cvidx = cv;
			foldIdx(fold).teidx = nets(~ismember(nets,[tr cv]));
		end
	end
end

