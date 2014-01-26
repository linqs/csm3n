function ex = readExample(prefix)
%
% Loads and formats an example from the Political Blog data
%
% prefix : file prefix, e.g., 'Feb_1'

Ynode = csvread([prefix '.Y.csv']);
nNode = size(Ynode,1);
Y = ones(nNode,1) + (Ynode(:,2) == 1);
Y = int32(Y);

edgeEnds = csvread([prefix '.Link.csv']);
nEdge = size(edgeEnds,1);
G = sparse(edgeEnds(:,1),edgeEnds(:,2),ones(nEdge,1),nNode,nNode);
G = G - diag(diag(G));
G = G | G';
edgeStruct = UGM_makeEdgeStruct(G,2,1);

[Aeq,beq] = pairwiseConstraints(edgeStruct);

Xnode = csvread([prefix '.Word.csv']);
nFeat = size(Xnode,2);
Xnode = reshape(Xnode',1,nFeat,nNode);
Xedge = makeEdgeFeatures(Xnode,edgeStruct.edgeEnds);

ex = makeExample(Xnode,Xedge,Y,2,edgeStruct,Aeq,beq);

