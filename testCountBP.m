
load data/testCountBP.mat

%% Bethe counts
clearvars -except ex w

edgeStruct = ex.edgeStruct;
[edgeStruct.nodeCount,edgeStruct.edgeCount] = UGM_BetheCounts(edgeStruct);
[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex.Xnode,ex.Xedge,ex.nodeMap,ex.edgeMap,edgeStruct);

% LBP
[nodeBel1,edgeBel1,logz1,h1] = UGM_Infer_LBP(nodePot,edgePot,edgeStruct);

% CountBP matlab
edgeStruct.useMex = 0;
% edgeStruct.maxIter = 1;
[nodeBel2,edgeBel2,logz2,h2] = UGM_Infer_CountBP(nodePot,edgePot,edgeStruct);
[sum(abs(nodeBel1(:)-nodeBel2(:))) sum(abs(edgeBel1(:)-edgeBel2(:))) logz1-logz2 h1-h2]

% CountBP mex
edgeStruct.useMex = 1;
[nodeBel3,edgeBel3,logz3,h3] = UGM_Infer_CountBP(nodePot,edgePot,edgeStruct);
[sum(abs(nodeBel1(:)-nodeBel3(:))) sum(abs(edgeBel1(:)-edgeBel3(:))) logz1-logz3 h1-h3]
% Verifies that mex & matlab agree
[sum(abs(nodeBel2(:)-nodeBel3(:))) sum(abs(edgeBel2(:)-edgeBel3(:))) logz2-logz3 h2-h3]


%% TRBP Counts
clearvars -except ex w

edgeStruct = ex.edgeStruct;
% edgeStruct.edgeDist = 0.01*ones(edgeStruct.nEdges,1);
[edgeStruct.nodeCount,edgeStruct.edgeCount] = UGM_TRBPCounts(edgeStruct);
[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex.Xnode,ex.Xedge,ex.nodeMap,ex.edgeMap,edgeStruct);

% TRBP
[nodeBel1,edgeBel1,logz1,h1] = UGM_Infer_TRBP(nodePot,edgePot,edgeStruct);

% CountBP matlab
edgeStruct.useMex = 0;
[nodeBel2,edgeBel2,logz2,h2] = UGM_Infer_CountBP(nodePot,edgePot,edgeStruct);
[sum(abs(nodeBel1(:)-nodeBel2(:))) sum(abs(edgeBel1(:)-edgeBel2(:))) logz1-logz2 h1-h2]

% CountBP mex
edgeStruct.useMex = 1;
[nodeBel3,edgeBel3,logz3,h3] = UGM_Infer_CountBP(nodePot,edgePot,edgeStruct);
[sum(abs(nodeBel1(:)-nodeBel3(:))) sum(abs(edgeBel1(:)-edgeBel3(:))) logz1-logz3 h1-h3]
% Verifies that mex & matlab agree
[sum(abs(nodeBel2(:)-nodeBel3(:))) sum(abs(edgeBel2(:)-edgeBel3(:))) logz2-logz3 h2-h3]


%% Decoding
clearvars -except ex w

edgeStruct = ex.edgeStruct;
[edgeStruct.nodeCount,edgeStruct.edgeCount] = UGM_TRBPCounts(edgeStruct);
[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex.Xnode,ex.Xedge,ex.nodeMap,ex.edgeMap,edgeStruct);

% LBP
labels1 = UGM_Decode_TRBP(nodePot,edgePot,edgeStruct);

% CountBP matlab
edgeStruct.useMex = 0;
labels2 = UGM_Decode_CountBP(nodePot,edgePot,edgeStruct);
nnz(labels1~=labels2)

% CountBP mex
edgeStruct.useMex = 1;
labels3 = UGM_Decode_CountBP(nodePot,edgePot,edgeStruct);
nnz(labels1~=labels3)
nnz(labels2~=labels3)


