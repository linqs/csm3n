function ex = makeExample(Xnode, Xedge, Y, nState, edgeStruct, Aeq, beq)
%
%Makes an example structure.
%

[nodeMap,edgeMap] = UGM_makeCRFmaps(Xnode,Xedge,edgeStruct,0,1,1);
Fx = makeVCTSMmap(Xnode,Xedge,nodeMap,edgeMap);
oc = overcompletePairwise(Y, edgeStruct);
ex.nNode = length(Y);
ex.nState = nState;
ex.ocLocalScope = ex.nNode * ex.nState;
ex.edgeStruct = edgeStruct;
ex.Y = Y;
ex.oc = oc;
ex.Xnode = Xnode;
ex.Xedge = Xedge;
ex.nodeMap = nodeMap;
ex.edgeMap = edgeMap;
ex.Fx = Fx;
ex.suffStat = Fx * oc;
ex.Aeq = Aeq;
ex.beq = beq;
