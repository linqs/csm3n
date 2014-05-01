function [ex_high,ex_low] = conceptDrift(rescale,nStates,nSampHigh,nSampLow,eta,alpha,burnIn,makePlots)

if ~exist('eta','var') || isempty(eta)
	eta = 0.2;
end
if ~exist('alpha','var') || isempty(alpha)
	alpha = 10;
end
if ~exist('burnIn','var') || isempty(burnIn)
	burnIn = 5;
end
if ~exist('makePlots','var') || isempty(makePlots)
	makePlots = 1;
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

%% Noisy ground truth

noisyimg = abs(srcimg - (rand(size(srcimg)) < eta));
y_noisy = noisyimg(:) + 1;


%% High coupling, low signal

w_max = 1; w_min = .1;
W_loc = zeros(nStates,nStatesY);
W_rel = zeros(nStates,nStates,2*nStatesY);
W_loc(:,1) = linspace(w_max,w_min,nStates)';
W_loc(:,2) = linspace(w_min,w_max,nStates)';
for i = 1:4
	W_rel(:,:,i) = .8 * (w_min*ones(nStates) + (w_max-w_min)*eye(nStates));
end
w_high = [W_loc(:) ; W_rel(:)];

% Sample from CRF conditioned on image
edgeStruct = UGM_makeEdgeStruct(G,nStates,1,nSampHigh);
Ynode = zeros(1,nStatesY,nNodes);
for n = 1:nNodes
	Ynode(1,y(n),n) = 1;
end
Yedge = makeEdgeFeatures(Ynode(1,:,:),edgeStruct.edgeEnds);
[nodeMap,edgeMap] = UGM_makeCRFmaps(Ynode,Yedge,edgeStruct,0,1,1);
[nodePot,edgePot] = UGM_CRF_makePotentials(w_high,Ynode,Yedge,nodeMap,edgeMap,edgeStruct);
samp_high = UGM_Sample_VarMCMC(nodePot,edgePot,edgeStruct,burnIn,.25)';


%% Low coupling, high signal

% W_loc = zeros(nStates,nStatesY);
% W_rel = zeros(nStates,nStates,2*nStatesY);
% W_loc(:,1) = (1 + alpha*10) * linspace(w_max,w_min,nStates)';
% W_loc(:,2) = (1 + alpha*10) * linspace(w_min,w_max,nStates)';
% for i = 1:4
% 	offDiagVal = w_min + alpha*(w_max-w_min);
% 	W_rel(:,:,i) = .8 * (offDiagVal*ones(nStates) + (w_max-offDiagVal)*eye(nStates));
% end
% w_low = [W_loc(:) ; W_rel(:)];
w_low = [alpha*W_loc(:) ; (1/alpha)*W_rel(:)];

% Sample from CRF conditioned on noisy image
edgeStruct = UGM_makeEdgeStruct(G,nStates,1,nSampLow);
Ynode = zeros(1,nStatesY,nNodes);
for n = 1:nNodes
	Ynode(1,y_noisy(n),n) = 1;
end
Yedge = makeEdgeFeatures(Ynode(1,:,:),edgeStruct.edgeEnds);
[nodeMap,edgeMap] = UGM_makeCRFmaps(Ynode,Yedge,edgeStruct,0,1,1);
[nodePot,edgePot] = UGM_CRF_makePotentials(w_low,Ynode,Yedge,nodeMap,edgeMap,edgeStruct);
samp_low = UGM_Sample_VarMCMC(nodePot,edgePot,edgeStruct,burnIn,.25)';


%% Make examples

% Structural info for ground truth
edgeStruct = UGM_makeEdgeStruct(G,nStatesY,1);
edgeStruct.edgeDist = UGM_makeEdgeDistribution(edgeStruct,3,[nRows nCols]);
edgeStruct.nRows = nRows; edgeStruct.nCols = nCols;
% [edgeStruct.nodeCount,edgeStruct.edgeCount] = UGM_ConvexBetheCounts(edgeStruct,1);

ex_high = cell(nSampHigh,1);
for i = 1:nSampHigh
	Xnode = zeros(1,nStates,nNodes);
	for n = 1:nNodes
		Xnode(1,samp_high(n,i),n) = 1;
	end
	Xedge = makeEdgeFeatures(Xnode(1,:,:),edgeStruct.edgeEnds);
	ex_high{i} = makeExample(Xnode(1,:,:),Xedge,y,nStatesY,edgeStruct,[],[]);
end

ex_low = cell(nSampLow,1);
for i = 1:nSampLow
	Xnode = zeros(1,nStates,nNodes);
	for n = 1:nNodes
		Xnode(1,samp_low(n,i),n) = 1;
	end
	Xedge = makeEdgeFeatures(Xnode(1,:,:),edgeStruct.edgeEnds);
	ex_low{i} = makeExample(Xnode(1,:,:),Xedge,y_noisy,nStatesY,edgeStruct,[],[]);
end


%% Plot samples

if makePlots
	% Ground truth
	fig = figure();
	subplot(2,2,1);
	imagesc(srcimg);
	title('Ground Truth');
	colormap(gray);

	% Noisy image
	subplot(2,2,2);
	imagesc(noisyimg);
	title('Noisy Ground Truth');

	% Train
	subplot(2,2,3);
	imagesc(reshape(samp_high(:,end),nRows,nCols));
	title('High Coupling, Low Signal Sample');

	% Test
	subplot(2,2,4);
	imagesc(reshape(samp_low(:,end),nRows,nCols));
	title('Low Coupling, High Signal Sample');

	% figPos = get(fig,'Position');
	% figPos(3) = 3*figPos(3);
	% set(fig,'Position',figPos);
end

%% Use noisy observations to decode labels
% 
% % High coupling model
% Wmax = 1; Wmin = .2;
% Wloc = zeros(2,nObsStates);
% Wrel = zeros(2,2,2*nObsStates);
% Wloc(1,:) = linspace(Wmax,Wmin,nObsStates);
% Wloc(2,:) = linspace(Wmin,Wmax,nObsStates);
% for i = 1:2*nObsStates
% 	Wrel(:,:,i) = 1 * (Wmin*ones(2) + (Wmax-Wmin)*eye(2));
% end
% w_high = [Wloc(:) ; Wrel(:)];
% 
% % Low coupling model
% Wloc = zeros(2,nObsStates);
% Wrel = zeros(2,2,2*nObsStates);
% Wloc(1,:) = logspace(Wmax,Wmin,nObsStates);
% Wloc(2,:) = logspace(Wmin,Wmax,nObsStates);
% for i = 1:2*nObsStates
% 	Wrel(:,:,i) = 0 * (Wmin*ones(2) + (Wmax-Wmin)*eye(2));
% end
% w_low = [Wloc(:) ; Wrel(:)];
% 
% 
% trLabels = zeros(nNodes,nSamp);
% for i = 1:nSamp
% 	Xnode = zeros(1,nObsStates,nNodes);
% 	for n = 1:nNodes
% 		Xnode(1,trSamp(n,i),n) = 1;
% 	end
% 	Xedge = makeEdgeFeatures(Xnode(1,:,:),edgeStruct.edgeEnds);
% 	[nodeMapObs,edgeMapObs] = UGM_makeCRFmaps(Xnode,Xedge,edgeStruct,0,1,1);
% 	[nodePot,edgePot] = UGM_CRF_makePotentials(w_high,Xnode,Xedge,nodeMapObs,edgeMapObs,edgeStruct);
% 	trLabels(:,i) = UGM_Decode_LBP(nodePot,edgePot,edgeStruct);
% end
% 
% teLabels = zeros(nNodes,nSamp);
% for i = 1:nSamp
% 	Xnode = zeros(1,nObsStates,nNodes);
% 	for n = 1:nNodes
% 		Xnode(1,teSamp(n,i),n) = 1;
% 	end
% 	Xedge = makeEdgeFeatures(Xnode(1,:,:),edgeStruct.edgeEnds);
% 	[nodeMapObs,edgeMapObs] = UGM_makeCRFmaps(Xnode,Xedge,edgeStruct,0,1,1);
% 	[nodePot,edgePot] = UGM_CRF_makePotentials(w_low,Xnode,Xedge,nodeMapObs,edgeMapObs,edgeStruct);
% 	teLabels(:,i) = UGM_Decode_LBP(nodePot,edgePot,edgeStruct);
% end


