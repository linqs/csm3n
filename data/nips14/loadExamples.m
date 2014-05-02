function examples = loadExamples(nEx,noiseRate,rescale,makeEdgeDist,makeCounts,makePlots)
%
% Loads and processes an example image
%
% nEx : number of examples
% noiseRate : noise rate
% rescale : rescale rate (def: 1)
% makeEdgeDist : whether to make the edge distribution (def: 1)
% makeCounts : whether to make the convexified Bethe counting numbers (def: 1)
% makePlots : whether to plot the source and noisy images (def: 0)

if ~exist('rescale','var')
	rescale = 1;
end
if ~exist('makeEdgeDist','var')
	makeEdgeDist = 1;
end
if ~exist('makeCounts','var')
	makeCounts = 1;
end
if ~exist('makePlots','var')
	makePlots = 0;
end

% Load image and convert grayscale to BW
srcimg = imread('nips2014.bmp','bmp');
srcimg = rgb2gray(srcimg);
if rescale ~= 1
	srcimg = imresize(srcimg,rescale);
end
[nRows,nCols,~] = size(srcimg);
maxval = max(unique(srcimg));
srcimg = srcimg == maxval;

% Ground truth
Ynode = srcimg(:) + 1;
nNode = length(Ynode);

% Make nEx noisy observations
Xnode = zeros(nEx,2,nNode);
for i = 1:nEx
	noisyimg = abs(srcimg - (rand(size(srcimg)) < noiseRate));
% 	noisyimg = (srcimg*2-1 + noiseRate*randn(size(srcimg))) > 0;
	Xnode(i,:,:) = [noisyimg(:)==0 noisyimg(:)==1]';
end

% Plots
if makePlots
	fig = figure();
	subplot(1,2,1);
	imagesc(srcimg);
	subplot(1,2,2);
	imagesc(noisyimg); % plot last noisy observation
	colormap(gray);
	figPos = get(fig,'Position');
	figPos(3) = 2*figPos(3);
	set(fig,'Position',figPos);
end

% Structural data
G = latticeAdjMatrix4(nRows,nCols);
edgeStruct = UGM_makeEdgeStruct(G,2,1);
edgeStruct.nRows = nRows; edgeStruct.nCols = nCols;
if makeEdgeDist
	edgeStruct.edgeDist = UGM_makeEdgeDistribution(edgeStruct,3,[nRows nCols]);
end
if makeCounts
	[edgeStruct.nodeCount,edgeStruct.edgeCount] = UGM_ConvexBetheCounts(edgeStruct,1);
end

% Make examples
examples = cell(nEx,1);
for i = 1:nEx
	Xedge = makeEdgeFeatures(Xnode(i,:,:),edgeStruct.edgeEnds);
	examples{i} = makeExample(Xnode(i,:,:),Xedge,Ynode,2,edgeStruct,[],[]);
end
