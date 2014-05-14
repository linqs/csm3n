
load data/testCountBP.mat
clearvars -except ex w

%% LBP
edgeStruct = ex.edgeStruct;
[edgeStruct.nodeCount,edgeStruct.edgeCount] = UGM_BetheCounts(edgeStruct);
[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex.Xnode,ex.Xedge,ex.nodeMap,ex.edgeMap,edgeStruct);

[nodeBel1,edgeBel1,logz1,h1] = UGM_Infer_LBP(nodePot,edgePot,edgeStruct);

% edgeStruct.edgeDist = UGM_makeEdgeDistribution(edgeStruct,0);
% [nodeBel2,edgeBel2,logz2,h2] = UGM_Infer_TRBP(nodePot,edgePot,edgeStruct);
% disp([sum(abs(nodeBel1(:)-nodeBel2(:))) sum(abs(edgeBel1(:)-edgeBel2(:))) logz1-logz2 h1-h2])

[nodeBel2,edgeBel2,logz2,h2] = UGM_Infer_CountBP(nodePot,edgePot,edgeStruct);
disp([sum(abs(nodeBel1(:)-nodeBel2(:))) sum(abs(edgeBel1(:)-edgeBel2(:))) logz1-logz2 h1-h2])

edgeStruct.useMex = 0;
[nodeBel3,edgeBel3,logz3,h3] = UGM_Infer_CountBP(nodePot,edgePot,edgeStruct);
disp([sum(abs(nodeBel1(:)-nodeBel3(:))) sum(abs(edgeBel1(:)-edgeBel3(:))) logz1-logz3 h1-h3])
disp([sum(abs(nodeBel2(:)-nodeBel3(:))) sum(abs(edgeBel2(:)-edgeBel3(:))) logz2-logz3 h2-h3])



%% TRBP
edgeStruct = ex.edgeStruct;
[edgeStruct.nodeCount,edgeStruct.edgeCount] = UGM_TRBPCounts(edgeStruct);
[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex.Xnode,ex.Xedge,ex.nodeMap,ex.edgeMap,edgeStruct);

[nodeBel1,edgeBel1,logz1,h1] = UGM_Infer_TRBP(nodePot,edgePot,edgeStruct);

[nodeBel2,edgeBel2,logz2,h2] = UGM_Infer_CountBP(nodePot,edgePot,edgeStruct);
max(abs(nodeBel1(:)-nodeBel2(:)))
% plot(abs(nodeBel1(:)-nodeBel2(:)))
disp([sum(abs(nodeBel1(:)-nodeBel2(:))) sum(abs(edgeBel1(:)-edgeBel2(:))) logz1-logz2 h1-h2])

edgeStruct.useMex = 0;
[nodeBel3,edgeBel3,logz3,h3] = UGM_Infer_CountBP(nodePot,edgePot,edgeStruct);
max(abs(nodeBel1(:)-nodeBel3(:)))
% plot(abs(nodeBel1(:)-nodeBel3(:)))
% [nodeBel1(1:40,1:5) nodeBel3(1:40,1:5)]
disp([sum(abs(nodeBel1(:)-nodeBel3(:))) sum(abs(edgeBel1(:)-edgeBel3(:))) logz1-logz3 h1-h3])
disp([sum(abs(nodeBel2(:)-nodeBel3(:))) sum(abs(edgeBel2(:)-edgeBel3(:))) logz2-logz3 h2-h3])
