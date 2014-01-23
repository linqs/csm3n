function [f, sg] = dlmObj(w, ex, t, decodeFunc, C, varargin)


% Local variables
eps = 1 / sqrt(t);
nNode = ex.nNode;
nEdge = ex.nEdge;
nNodeFeat = ex.nNodeFeat;
nEdgeFeat = ex.nEdgeFeat;
nState = ex.nState;
Ynode = overcompleteRep(ex.Y,nState,0);
Xnode = ex.Xnode;
Xedge = ex.Xedge;
nodeMap = ex.nodeMap;
edgeMap = ex.edgeMap;
edgeStruct = ex.edgeStruct;


%% Inference

% Regular inference
[nodePot,edgePot] = UGM_CRF_makePotentials(w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct);
yhat = decodeFunc(nodePot,edgePot,edgeStruct,varargin{:});

% Loss-augmented inference
nodePot = nodePot .* exp(eps * (1 - Ynode'));
% nodePot = nodePot .* exp(eps * (Ynode' - 1));
ydir = decodeFunc(nodePot,edgePot,edgeStruct,varargin{:});


%% Compute objective/gradient

% L2 regularization
f = 0.5 * (C .* w)' * w;
sg = C .* w;

% Hamming loss
f = f + eps * nnz(ex.Y ~= ydir);
% f = f - eps * nnz(ex.Y ~= ydir);

% Overcomplete reps
yhat_oc = overcompletePairwise(yhat,edgeStruct);
ydir_oc = overcompletePairwise(ydir,edgeStruct);

% Nodes
widx = reshape(nodeMap(1,:,:),nState,nNodeFeat);
yidx = localIndex(1,1:nState,nState);
Wnode = w(widx);
for i = 1:nNode
	Fxy = (ydir_oc(yidx) - yhat_oc(yidx)) * Xnode(1,:,i);
% 	Fxy = (yhat_oc(yidx) - ydir_oc(yidx)) * Xnode(1,:,i);
	f = f + sum(sum(Wnode .* Fxy));
	sg(widx) = sg(widx) + Fxy;
	yidx = yidx + nState;
end

% Edges
widx = reshape(edgeMap(:,:,1,:),nState^2,nEdgeFeat);
yidx = pairwiseIndex(1,1:nState,1:nState,nNode,nState);
Wedge = w(widx);
for e = 1:nEdge
	Fxy = (ydir_oc(yidx) - yhat_oc(yidx)) * Xedge(1,:,e);
% 	Fxy = (yhat_oc(yidx) - ydir_oc(yidx)) * Xedge(1,:,e);
	f = f + sum(sum(Wedge .* Fxy));
	sg(widx) = sg(widx) + Fxy;
	yidx = yidx + nState^2;
end



