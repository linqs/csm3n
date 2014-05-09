function [examples] = entropicModel(rescale,nStates,nSamp,w_ratio,alpha,burnIn,plotFig)

if ~exist('w_ratio','var') || isempty(w_ratio)
	w_ratio = .1;
end
if ~exist('alpha','var') || isempty(alpha)
	alpha = .8;
end
if ~exist('burnIn','var') || isempty(burnIn)
	burnIn = 10;
end
if ~exist('plotFig','var') || isempty(plotFig)
	plotFig = 1;
end

%% Load image and convert grayscale to BW

srcimg = imread('nips2014.bmp','bmp');
srcimg = rgb2gray(srcimg);
if rescale ~= 1
	srcimg = imresize(srcimg,rescale);
end
[nRows,nCols,~] = size(srcimg);
G = latticeAdjMatrix4(nRows,nCols);
nNodes = nRows * nCols;
maxval = max(unique(srcimg));
srcimg = srcimg == maxval;
y = srcimg(:) + 1;
nStatesY = 2;


%% High coupling, low signal

w_max = 1;
w_min = w_ratio * w_max;
W_loc = zeros(nStates,nStatesY);
W_rel = zeros(nStates,nStates,2*nStatesY);
W_loc(:,1) = linspace(w_max,w_min,nStates)';
W_loc(:,2) = linspace(w_min,w_max,nStates)';
for i = 1:4
	W_rel(:,:,i) = alpha * (w_min*ones(nStates) + (w_max-w_min)*eye(nStates));
end
w_high = [W_loc(:) ; W_rel(:)];

% Sample from CRF conditioned on image
edgeStruct = UGM_makeEdgeStruct(G,nStates,1,nSamp);
Ynode = zeros(1,nStatesY,nNodes);
for n = 1:nNodes
	Ynode(1,y(n),n) = 1;
end
Yedge = makeEdgeFeatures(Ynode(1,:,:),edgeStruct.edgeEnds);
[nodeMap,edgeMap] = UGM_makeCRFmaps(Ynode,Yedge,edgeStruct,0,1,1);
[nodePot,edgePot] = UGM_CRF_makePotentials(w_high,Ynode,Yedge,nodeMap,edgeMap,edgeStruct);
samp = UGM_Sample_VarMCMC(nodePot,edgePot,edgeStruct,burnIn,.25)';


%% Make examples

% Structural info for ground truth
edgeStruct = UGM_makeEdgeStruct(G,nStatesY,1);
edgeStruct.edgeDist = UGM_makeEdgeDistribution(edgeStruct,3,[nRows nCols]);
edgeStruct.nRows = nRows; edgeStruct.nCols = nCols;
% [edgeStruct.nodeCount,edgeStruct.edgeCount] = UGM_ConvexBetheCounts(edgeStruct,1,.1);

examples = cell(nSamp,1);
for i = 1:nSamp
	Xnode = zeros(1,nStates,nNodes);
	for n = 1:nNodes
		Xnode(1,samp(n,i),n) = 1;
	end
	Xedge = makeEdgeFeatures(Xnode(1,:,:),edgeStruct.edgeEnds);
	examples{i} = makeExample(Xnode(1,:,:),Xedge,y,nStatesY,edgeStruct,[],[]);
end


%% Plot samples

if plotFig
	% Ground truth
	fig = figure(plotFig);
	subplot(1,2,1);
	imagesc(srcimg);
	title('Ground Truth');
	colormap(gray);

	% Samples
	subplot(1,2,2);
	imagesc(reshape(samp(:,end),nRows,nCols));
	title('Samples');

% 	figPos = get(fig,'Position');
% 	figPos(3) = 2*figPos(3);
% 	set(fig,'Position',figPos);
end

