function examples = loadGrabCut(makeEdgeDist,nEx, countBP, validity, scaled)

if nargin < 1
    makeEdgeDist = 1;
end
if nargin < 4
    validity = 0;
end
if nargin < 5
    scaled = 0;
end

if scaled
    load grabCutProcessed;
else
    load grabCutProcessedFull;
end

if nargin < 2 || nEx > length(images)
    nEx = length(images);
end

examples = cell(nEx,1);

MIN_FEATURE = -10;

totalTimer = tic;

for i = 1:nEx
    
    [nRows, nCols] = size(label{i});
    
    %% STRUCTURAL VARIABLES
    nNode = nRows * nCols;
    nState = 2;
    [G, edgeDirections] = latticeAdjMatrix8(nRows,nCols);
    
    % trim away known variables
    mask = trimap{i} == 128;
    G = G(mask(:), mask(:));
    edgeDirections = edgeDirections(mask(:), mask(:));
    
    edgeStruct = UGM_makeEdgeStruct(G,nState,1);
    nEdge = edgeStruct.nEdges;
    if makeEdgeDist
        edgeStruct.edgeDist = UGM_makeEdgeDistribution(edgeStruct,3,[nRows nCols]);
    end
    
    
    if countBP
        kappa = 1;
        minKappa = 0.01;
        [edgeStruct.nodeCount,edgeStruct.edgeCount] = UGM_ConvexBetheCounts(edgeStruct,kappa,minKappa, validity);
    end
    
    %% FEATURES
    
    % Node features are GMM posterior probabilities
    Xnode = zeros(1, 0, nnz(mask));
    Xnode(1,end+1,:) = (max(MIN_FEATURE, log(probObj{i}(mask))) - MIN_FEATURE) / abs(MIN_FEATURE);
    Xnode(1,end+1,:) = (max(MIN_FEATURE, log(1-probObj{i}(mask))) - MIN_FEATURE) / abs(MIN_FEATURE);
    % Xnode(1,end+1,:) = (max(MIN_FEATURE, log(probObject{i}(mask))) - MIN_FEATURE)/ abs(MIN_FEATURE);
    % Xnode(1,end+1,:) = (max(MIN_FEATURE, log(probBackgr{i}(mask))) - MIN_FEATURE)/ abs(MIN_FEATURE);
    
% %     masks = { [1 1 1], [1 1 1]', [0 0 1; 0 1 0; 1 0 0], [1 0 0; 0 1 0; 0 0 1] };
     masks = {ones(3)};
    
    for m = 1:length(masks)
        border = (conv2(double(trimap{i} == 64), masks{m}, 'same') > 0);
        Xnode(1,end+1,:) = border(mask);
        border = (conv2(double(trimap{i} == 255), masks{m}, 'same') > 0);
        Xnode(1,end+1,:) = border(mask);
    end
    
    
    
    Xnode(1,end+1,:) = 1; % bias feature
    
    %% Edge features are RBF between pixel intensities
    X_bw = double(rgb2gray(images{i}));
    X_bw = X_bw(mask);
    
    Xedge = makeRbfEdgeFeatures(edgeStruct.edgeEnds, X_bw, edgeDirections, 1/255);
    
    % check edge features
    
    maskIdx = find(mask);
    for d = 1:4
        clf;
        subplot(211);
        imagesc(images{i});
        subplot(212);
        
        img = sparse(maskIdx(edgeStruct.edgeEnds(:,1)), ...
            ones(edgeStruct.nEdges,1), squeeze(Xedge(1,d,:)), nRows*nCols, 1);
        
        img = reshape(img, nRows, nCols);
        
        imagesc(img);
        title(sprintf('Edge feature %d', d));
        colormap gray;
        drawnow;
    end
    
    
    %% LABELS
    
    y = 2 - (label{i}(mask) > 128);
    
    %% MAKE EXAMPLE
    % [Aeq,beq] = pairwiseConstraints(edgeStruct);
    Aeq = []; beq = [];
    ex = makeExample(Xnode,Xedge,y,nState,edgeStruct,Aeq,beq);
    ex.srcrgb = images{i};
    ex.srcbw = double(rgb2gray(images{i}));
    ex.probObj = probObj{i};
    ex.mask = mask;
    
    examples{i} = ex;
    
    %% plot node features
    
    figRows = 5;
    figCols = ceil(size(Xnode,2) / figRows);
    
    clf;
    for d = 1:size(Xnode,2)
        img = 0.5*ones(nRows, nCols);
        
        img(mask) = Xnode(1,d,:);
        
        subplot(figRows, figCols, d);
        imagesc(img);
        colormap gray;
        title(sprintf('Feature %d', d));
        
    end
    drawnow;
    
    %%
    
    fprintf('Finished setting up %d of %d examples (%d nodes, %d edges) in %2.2f minutes. ETA %2.2f minutes\n', ...
        i, nEx, ex.nNode, ex.nEdge, toc(totalTimer)/60, (toc(totalTimer)/60/i) * (nEx - i));
end
