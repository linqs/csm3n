function ex = makeExample(Xnode, Xedge, y, nState, edgeStruct, Aeq, beq)
%
% Makes an example structure.
%

[nodeMap,edgeMap] = UGM_makeCRFmaps(Xnode,Xedge,edgeStruct,0,1,1);
ex.nodeMap = nodeMap;
ex.edgeMap = edgeMap;
ex.nNode = length(y);
ex.nEdge = edgeStruct.nEdges;
ex.nNodeFeat = size(Xnode,2);
ex.nEdgeFeat = size(Xedge,2);
ex.nState = nState;
ex.ocLocalScope = ex.nNode * nState;
ex.edgeStruct = edgeStruct;
ex.oc = overcompletePairwise(y,nState,edgeStruct);
ex.Y = int32(y);
ex.Ynode = overcompleteRep(y,nState,0);
ex.Xnode = Xnode;
ex.Xedge = Xedge;
ex.Fx = makeFx(Xnode,Xedge,nodeMap,edgeMap);
ex.suffStat = ex.Fx * ex.oc;
ex.Aeq = Aeq;
ex.beq = beq;
