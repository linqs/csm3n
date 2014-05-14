
load data/testCountBP.mat
clearvars -except ex w

%% Bethe counts

edgeStruct = ex.edgeStruct;
[edgeStruct.nodeCount,edgeStruct.edgeCount] = UGM_BetheCounts(edgeStruct);
[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex.Xnode,ex.Xedge,ex.nodeMap,ex.edgeMap,edgeStruct);

% LBP
[nodeBel1,edgeBel1,logz1,h1] = UGM_Infer_LBP(nodePot,edgePot,edgeStruct);

% CountBP mex
[nodeBel2,edgeBel2,logz2,h2] = UGM_Infer_CountBP(nodePot,edgePot,edgeStruct);
[sum(abs(nodeBel1(:)-nodeBel2(:))) sum(abs(edgeBel1(:)-edgeBel2(:))) logz1-logz2 h1-h2]

% CountBP matlab
edgeStruct.useMex = 0;
[nodeBel3,edgeBel3,logz3,h3] = UGM_Infer_CountBP(nodePot,edgePot,edgeStruct);
[sum(abs(nodeBel1(:)-nodeBel3(:))) sum(abs(edgeBel1(:)-edgeBel3(:))) logz1-logz3 h1-h3]
% [sum(abs(nodeBel2(:)-nodeBel3(:))) sum(abs(edgeBel2(:)-edgeBel3(:))) logz2-logz3 h2-h3]

% GBP matlab
edgeStruct.useMex = 0;
[nodeBel4,edgeBel4,logz4,h4] = UGM_Infer_GBP(nodePot,edgePot,edgeStruct);
[sum(abs(nodeBel1(:)-nodeBel4(:))) sum(abs(edgeBel1(:)-edgeBel4(:))) logz1-logz4 h1-h4]
% [sum(abs(nodeBel4(:)-nodeBel3(:))) sum(abs(edgeBel4(:)-edgeBel3(:))) logz4-logz3 h4-h3]


%% TRBP Counts

edgeStruct = ex.edgeStruct;
edgeStruct.edgeDist = 0.01*ones(edgeStruct.nEdges,1);
[edgeStruct.nodeCount,edgeStruct.edgeCount] = UGM_TRBPCounts(edgeStruct);
% edgeStruct.nodeCount = edgeStruct.nodeCount / .01;
[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex.Xnode,ex.Xedge,ex.nodeMap,ex.edgeMap,edgeStruct);

% TRBP
[nodeBel1,edgeBel1,logz1,h1] = UGM_Infer_TRBP(nodePot,edgePot,edgeStruct);

% CountBP mex
[nodeBel2,edgeBel2,logz2,h2] = UGM_Infer_CountBP(nodePot,edgePot,edgeStruct);
max(abs(nodeBel1(:)-nodeBel2(:)))
% plot(abs(nodeBel1(:)-nodeBel2(:)))
[sum(abs(nodeBel1(:)-nodeBel2(:))) sum(abs(edgeBel1(:)-edgeBel2(:))) logz1-logz2 h1-h2]

% CountBP matlab
edgeStruct.useMex = 0;
[nodeBel3,edgeBel3,logz3,h3] = UGM_Infer_CountBP(nodePot,edgePot,edgeStruct);
max(abs(nodeBel1(:)-nodeBel3(:)))
nnz(abs(nodeBel1-nodeBel3) * ones(edgeStruct.nStates(1),1) > .1)
% plot(abs(nodeBel1(:)-nodeBel3(:)))
% [nodeBel1(1:40,1:5) nodeBel3(1:40,1:5)]
[sum(abs(nodeBel1(:)-nodeBel3(:))) sum(abs(edgeBel1(:)-edgeBel3(:))) logz1-logz3 h1-h3]
% [sum(abs(nodeBel2(:)-nodeBel3(:))) sum(abs(edgeBel2(:)-edgeBel3(:))) logz2-logz3 h2-h3]

% GBP matlab
edgeStruct.useMex = 0;
[nodeBel4,edgeBel4,logz4,h4] = UGM_Infer_GBP(nodePot,edgePot,edgeStruct);
[sum(abs(nodeBel1(:)-nodeBel4(:))) sum(abs(edgeBel1(:)-edgeBel4(:))) logz1-logz4 h1-h4]
% [sum(abs(nodeBel4(:)-nodeBel3(:))) sum(abs(edgeBel4(:)-edgeBel3(:))) logz4-logz3 h4-h3]

% SBP matlab
edgeStruct.useMex = 0;
edgeStruct.auxCount = 0.5 * [edgeStruct.edgeDist ; edgeStruct.edgeDist];
nodeBel4 = UGM_Infer_SBP(nodePot,edgePot,edgeStruct);
sum(abs(nodeBel1(:)-nodeBel4(:)))


