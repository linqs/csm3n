function examples = loadGrabCut(makeEdgeDist,nEx)

if nargin < 1
	makeEdgeDist = 1;
end

load grabCutProcessedSmall;

if nargin < 2 || nEx > length(images)
	nEx = length(images);
end

examples = cell(nEx,1);

MIN_FEATURE = -20;

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
    
    
    %% FEATURES
    
    % Node features are GMM posterior probabilities
    Xnode = zeros(1, 4, nnz(mask));
    Xnode(1,1,:) = max(MIN_FEATURE, log(probObj{i}(mask)));
    Xnode(1,2,:) = max(MIN_FEATURE, log(1-probObj{i}(mask)));
    border = (conv2(double(trimap{i} == 64), ones(3), 'same') > 0)*2-1;
    Xnode(1,3,:) = border(mask);
    border = (conv2(double(trimap{i} == 255), ones(3), 'same') > 0)*2-1;
    Xnode(1,4,:) = border(mask);
  
    % Edge features are RBF between pixel intensities
    X_bw = double(rgb2gray(images{i}));
    X_bw = X_bw(mask);
    
    Xedge = makeRbfEdgeFeatures(edgeStruct.edgeEnds, X_bw, edgeDirections, 50);
    
    %% LABELS
    
    y = 2 - (label{i}(mask) > 128);
    
    %% MAKE EXAMPLE
    % [Aeq,beq] = pairwiseConstraints(edgeStruct);
    Aeq = []; beq = [];
    ex = makeExample(Xnode,Xedge,y,nState,edgeStruct,Aeq,beq);
    ex.srcrgb = images{i};
    ex.srcbw = double(rgb2gray(images{i}));
    ex.probObject = probObject{i};
    ex.probBackgr = probBackgr{i};
    ex.mask = mask;
    
    examples{i} = ex;
    
    %% plot node features
    
    clf;
    for d = 1:size(Xnode,2)
        img = zeros(nRows, nCols);
        
        img(mask) = Xnode(1,d,:);
        
        subplot(size(Xnode,2), 1, d);
        imagesc(img);
        colormap gray;
        title(sprintf('Feature %d', d));
    end
    %%

    fprintf('Finished setting up %d of %d examples in %2.2f minutes. ETA %2.2f minutes\n', i, nEx, toc(totalTimer)/60, (toc(totalTimer)/60/i) * (nEx - i));
end
