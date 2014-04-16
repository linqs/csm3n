function [examples, foldIdx] = loadDocDataSnowball(fName, nNet, nPC, jumpRate, makeEdgeDist, plotNets)

if ~exist('jumpRate','var') || isempty(jumpRate)
	jumpRate = 0;
end
if ~exist('makeEdgeDist','var') || isempty(makeEdgeDist)
	makeEdgeDist = 0;
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
subgraphs = snowballSample(G,nNet,jumpRate,[],plotNets);

% Perform PCA on all observed features
X = bsxfun(@minus,full(X),mean(X,1));
[V,~] = eigs(X'*X / (size(X,1)-1), nPC);
X = X * V;

% Create nNet network examples
examples = cell(nNet,1);
for i = 1:nNet
	
	edgeStruct = UGM_makeEdgeStruct(subgraphs(i).A,nState,1);
% 	[Aeq,beq] = pairwiseConstraints(edgeStruct);
	Aeq = []; beq = [];

	if makeEdgeDist
		edgeStruct.edgeDist = UGM_makeEdgeDistribution(edgeStruct);
	end
	
	Xnode = zeros(1,nPC,length(subgraphs(i).nodes));
	Xnode(1,:,:) = X(subgraphs(i).nodes,:)';
	Xedge = makeEdgeFeatures(Xnode,edgeStruct.edgeEnds);
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

