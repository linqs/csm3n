function ex = readExample(fname,makeEdgeDist)


if nargin < 2
	makeEdgeDist = 0;
end


%% OBSERVED PIXELS

% Load RGB image
srcrgb = imread(sprintf('%s/src_color/%s.png',fname,fname),'png');
[nRows,nCols,~] = size(srcrgb);
X_c = double(reshape(srcrgb,[],3)) / 255;

% Load BW image
srcbw = imread(sprintf('%s/src_bw/%s.png',fname,fname),'png');
X_bw = double(reshape(srcbw(:,:,1),[],1)) / 255;


%% STRUCTURAL VARIABLES
nNode = nRows * nCols;
nState = 2;
G = latticeAdjMatrix4(nRows,nCols);
edgeStruct = UGM_makeEdgeStruct(G,nState,1);
nEdge = edgeStruct.nEdges;
if makeEdgeDist
	edgeStruct.edgeDist = UGM_makeEdgeDistribution(edgeStruct,3,[nRows nCols]);
end


%% FEATURES

% Fit a 2-mean GMM to the pixel RGB
gmm = gmdistribution.fit(X_c,nState);
p = posterior(gmm,X_c);

% Node features are GMM posterior probabilities
Xnode(1,:,:) = p';

% Edge features are concatenations of Xnode & RBF between pixel intensities
Xedge = makeEdgeFeatures(Xnode,edgeStruct.edgeEnds,X_bw);


%% LABELS

% Load labelings
fnames = dir(sprintf('%s/human_seg/*.png',fname));
humansegs = zeros(nRows*nCols,3);
for f = 1:3
	seg = imread(sprintf('%s/human_seg/%s',fname,fnames(f).name),'png');
	seg = reshape(seg,nNode,3);
	% Interpret red (255,0,0) as 1; otherwise, 0
	humansegs(:,f) = (seg(:,1) == 255) & (seg(:,2) == 0) & (seg(:,3) == 0);
end

% Take majority vote as label
y = sum(humansegs,2) >= 2;
y = int32(y+1);


%% MAKE EXAMPLE
% [Aeq,beq] = pairwiseConstraints(edgeStruct);
Aeq = []; beq = [];
ex = makeExample(Xnode,Xedge,y,nState,edgeStruct,Aeq,beq);
ex.gmm = gmm;
ex.srcbw = srcbw;

