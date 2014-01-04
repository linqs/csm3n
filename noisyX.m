function examples = noisyX(dispPlot)
%
% Creates a "noisy X" image segmentation dataset, per Mark Schmidt.
%
% dispPlot : whether to plot first 2 examples

if nargin < 1
	dispPlot = 0;
end

load Ximage.mat
[nRows,nCols] = size(Ximage);
nNode = nRows * nCols;
nEx = 10;
nStateY = 2;

% Make noisy X instances
Y = reshape(Ximage,[nNode 1]);
Y = repmat(Y,[1 nEx]);
Y = Y + 0.25*randn(size(Y));
Y = int32(Y > 0.5);
Y = Y + 1;

% noisy observations
obs = double(Y) + 0.5*randn(size(Y));
obs = obs / 4;
obs(obs > 1) = 1;

% discretize observations
discretize = 0;
if discretize
	nStateX = 4;
	obs = round(obs * nStateX);
	obs(obs < 1) = 1;
	X = zeros(nEx,nStateX,nNode);
	for i = 1:nEx
		X(i,:,:) = overcompleteRep(obs(:,i),nStateX,0);
	end
else
	X = zeros(nEx,1,nNode);
	for i = 1:nEx
		X(i,:,:) = obs(:,i)';
	end
end

% plot some examples
if dispPlot
	figure;
	for i = 1:2
		subplot(2,2,i);
		imagesc(reshape(Y(:,i),nRows,nCols));
		subplot(2,2,i+2);
		imagesc(reshape(obs(:,i),nRows,nCols));
		colormap gray
	end
	suptitle('Examples of Noisy Xs');
end

% adjacency graph
G = latticeAdjMatrix(nRows,nCols);

% convert to cell array of examples
examples = cell(nEx,1);
edgeStruct = UGM_makeEdgeStruct(G,nStateY,1);
[Aeq,beq] = pairwiseConstraints(edgeStruct);
for i = 1:nEx
	Xnode = X(i,:,:);
	Xedge = UGM_makeEdgeFeatures(Xnode,edgeStruct.edgeEnds);
	[nodeMap,edgeMap] = UGM_makeCRFmaps(Xnode,Xedge,edgeStruct,0,1,1);
	F = makeVCTSMmap(Xnode,Xedge,nodeMap,edgeMap);
	examples{i}.nNode = nNode;
	examples{i}.nState = nStateY;
	examples{i}.G = G;
	examples{i}.Y = Y(:,i);
	examples{i}.edgeStruct = edgeStruct;
	examples{i}.Xnode = Xnode;
	examples{i}.Xedge = Xedge;
	examples{i}.nodeMap = nodeMap;
	examples{i}.edgeMap = edgeMap;
	examples{i}.F = F;
	examples{i}.suffStat = F * overcompletePairwise(Y(:,i), edgeStruct);
	examples{i}.Aeq = Aeq;
	examples{i}.beq = beq;
end

% % check constraints using ground truth
% err = 0;
% for i = 1:nEx
% 	y = overcompletePairwise(examples{i}.Y, edgeStruct);
% 	err = err + sum(Aeq * y - beq);
% end
% err


