function examples = noisyX(obsNoise, dispPlot)
%
% Creates a "noisy X" image segmentation dataset, per Mark Schmidt.
%
% obsNoise : noise rate for observed variables
% dispPlot : whether to plot first 2 examples

if nargin < 1
	obsNoise = 0.5;
end
if nargin < 2
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
% Y = Y + 0.25*randn(size(Y));
Y = int32(Y > 0.5);
Y = Y + 1;

% noisy observations
obs = (double(Y) - 1) * 2 - 1 + obsNoise * randn(size(Y));
X = zeros(nEx,1,nNode);
for i = 1:nEx
	X(i,:,:) = obs(:,i)';
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
	examples{i} = makeExample(Xnode,Xedge,Y(:,i),nStateY,edgeStruct,Aeq,beq);
end

