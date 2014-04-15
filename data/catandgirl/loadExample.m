function ex = loadExample(fname,noiseRate,makeEdgeDist,makeCounts,makePlots)
%
% Loads and processes an example image
%
% fname : file name
% noiseRate : noise rate
% makeEdgeDist : whether to make the edge distribution (def: 1)
% makeCounts : whether to make the convexified Bethe counting numbers (def: 0)
% makePlots : whether to plot the source and noisy images (def: 0)

if nargin < 3
	makeEdgeDist = 1;
end
if nargin < 4
	makeCounts = 0;
end
if nargin < 5
	makePlots = 0;
end

% Load image and convert grayscale to BW
srcimg = imread(fname,'gif');
[nRows,nCols,~] = size(srcimg);
maxval = max(unique(srcimg));
srcimg = srcimg == maxval;

% Ground truth
Ynode = srcimg(:) + 1;
nNode = length(Ynode);

% Noisy observation
noisyimg = abs(srcimg - (rand(size(srcimg)) < noiseRate));
Xnode = zeros(1,2,nNode);
Xnode(1,:,:) = [noisyimg(:)==0 noisyimg(:)==1]';

% Plots
if makePlots
	subplot(1,2,1);
	imagesc(srcimg);
	subplot(1,2,2);
	imagesc(noisyimg,[0 1]);
	colormap(gray);
end

% Structural data
G = latticeAdjMatrix4(nRows,nCols);
edgeStruct = UGM_makeEdgeStruct(G,2,1);
if makeEdgeDist
	edgeStruct.edgeDist = UGM_makeEdgeDistribution(edgeStruct,3,[nRows nCols]);
end
if makeCounts
	[edgeStruct.nodeCount,edgeStruct.edgeCount] = UGM_ConvexBetheCounts(edgeStruct,1);
end

% Edge features
Xedge = makeEdgeFeatures(Xnode,edgeStruct.edgeEnds);

% Make example struct
ex = makeExample(Xnode,Xedge,Ynode,2,edgeStruct,[],[]);

