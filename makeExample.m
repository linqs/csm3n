function ex = makeExample(Xnode, Xedge, Y, nState, edgeStruct, Aeq, beq)
%
% Makes an example structure.
%

[nodeMap,edgeMap] = UGM_makeCRFmaps(Xnode,Xedge,edgeStruct,0,1,1);
ex.nodeMap = nodeMap;
ex.edgeMap = edgeMap;
ex.nNode = length(Y);
ex.nEdge = edgeStruct.nEdges;
ex.nNodeFeat = size(Xnode,2);
ex.nEdgeFeat = size(Xedge,2);
ex.nState = nState;
ex.ocLocalScope = ex.nNode * ex.nState;
ex.edgeStruct = edgeStruct;
ex.oc = overcompletePairwise(Y, edgeStruct);
ex.Y = int32(Y);
ex.Ynode = reshape(ex.oc(1:ex.ocLocalScope),nState,ex.nNode);
ex.Xnode = Xnode;
ex.Xedge = Xedge;
ex.Fx = makeVCTSMmap(Xnode,Xedge,nodeMap,edgeMap);
ex.suffStat = ex.Fx * ex.oc;
ex.Aeq = Aeq;
ex.beq = beq;
