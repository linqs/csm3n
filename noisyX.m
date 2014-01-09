function examples = noisyX(nEx, obsNoise, addBias, dispPlot)
%
% Creates a "noisy X" image segmentation dataset, per Mark Schmidt.
%
% nEx : number of examples to generate
% obsNoise : noise rate for observed variables
% dispPlot : whether to plot first 2 examples

if nargin < 2
	obsNoise = 0.5;
end
if nargin < 3
	addBias = 0;
end
if nargin < 4
	dispPlot = 0;
end

load Ximage.mat
[nRows,nCols] = size(Ximage);
nNode = nRows * nCols;
nStateY = 2;

% Make noisy X instances
Y = reshape(Ximage,[nNode 1]);
Y = repmat(Y,[1 nEx]);
% Y = Y + 0.25*randn(size(Y));
Y = int32(Y > 0.5);
Y = Y + 1;

% noisy observations
obs = (double(Y) - 1) * 2 - 1 + obsNoise * randn(size(Y));
X = zeros(nEx,1+addBias,nNode);
for i = 1:nEx
	if addBias
		X(i,:,:) = [ones(1,nNode) ; obs(:,i)'];
	else
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
	examples{i} = makeExample(Xnode,Xedge,Y(:,i),nStateY,edgeStruct,Aeq,beq);
end

