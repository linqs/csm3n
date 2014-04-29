function [examples, foldIdx] = loadDocData(fName, nNet, splits, nPC, makeEdgeDist, plotNets)

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

% Perform PCA on all observed features
X = bsxfun(@minus,full(X),mean(X,1));
[V,~] = eigs(X'*X / (size(X,1)-1), nPC);
X = X * V;

% Compute splits
if isempty(splits)
	nNode = floor(length(y) / nNet);
	splits = cell(nNet);
	for i = 1:nNet
		if i < nNet || mod(length(y),nNet) == 0
			splits{i} = (i-1)*nNode+1:i*nNode;
		else
			splits{i} = (i-1)*nNode+1:length(y);
		end
	end
end
% % cora
% splits = {1:750,751:1750,1751:length(y)};
% % citeseer
% splits = {1:1000,1001:2200,2200:length(y)};

% Partition into nNet networks
examples = cell(nNet,1);
for i = 1:nNet
	
	idx = splits{i};
	
	if plotNets
		subplot(1,nNet,i);
		spy(G(idx,idx));
	end
	
	edgeStruct = UGM_makeEdgeStruct(G(idx,idx),nState,1);
% 	[Aeq,beq] = pairwiseConstraints(edgeStruct);
	Aeq = []; beq = [];
	
	if makeEdgeDist
		edgeStruct.edgeDist = UGM_makeEdgeDistribution(edgeStruct);
	end
	
	Xnode = zeros(1,nPC,length(idx));
	Xnode(1,:,:) = X(idx,:)';
	Xedge = makeEdgeFeatures(Xnode,edgeStruct.edgeEnds);
	examples{i} = makeExample(Xnode,Xedge,y(idx),nState,edgeStruct,Aeq,beq);

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

% % Generate nNet! folds of all permutations of 1:nNet, with
% %  1 train
% %  1 unlabeled (if nNet >= 4)
% %  1 cv
% %  rest are test
% exPerms = perms(1:nNet);
% for fold = 1:size(exPerms,1)
% 	foldIdx(fold).tridx = exPerms(fold,1);
% 	if nNet >= 4
% 		foldIdx(fold).ulidx = exPerms(fold,2);
% 		foldIdx(fold).cvidx = exPerms(fold,3);
% 		foldIdx(fold).teidx = exPerms(fold,4:end);
% 	else
% 		foldIdx(fold).cvidx = exPerms(fold,2);
% 		foldIdx(fold).teidx = exPerms(fold,3);
% 	end
% end


% OLD CODE, DO NOT USE!!! Will exhaust RAM.
% % Perform PCA for every group of nNet-1 networks, 
% %	simulating inductive learning.
% % nPC is the number of principle components to use.
% % examples{i,j} contains example network j when example network
% %	i is held out for testing.
% nPC = 100;
% examples = cell(nNet);
% pcs = cell(nNet);
% for i = 1:nNet
% 	idx = [];
% 	for j = 1:nNet
% 		if j ~= i
% 			idx = [idx ; nets(j).idx(:)];
% 		end
% 	end
% 	X_not_i = full(X(idx,:));
% 	featAvg = mean(X_not_i,1);
% 	projMat = princomp(X_not_i);
% 	for j = 1:nNet
% 		Xnode(1,:,:) = (bsxfun(@minus,nets(j).X,featAvg) * projMat(:,1:nPC))';
% 		Xedge = makeEdgeFeatures(Xnode,nets(j).edgeStruct.edgeEnds);
% 		examples{i,j} = makeExample(Xnode,Xedge,nets(j).y,nState,...
% 							nets(j).edgeStruct,nets(j).Aeq,nets(j).beq);
% 	end
% end
% 

