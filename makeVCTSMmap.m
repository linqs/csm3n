function F = makeVCTSMmap(Xnode, Xedge, nodeMap, edgeMap)
%
% Creates the feature map for VCTSM learning.
%
% Xnode : 1 x nNodeFeat x nNode matrix of observed node features.
% Xedge : 1 x nEdgeFeat x nEdge matrix of observed edge features.
% nodeMap : UGM parameter map for nodes
% edgeMap : UGM parameter map for edges

% constants
[nNode,nState,nNodeFeat] = size(nodeMap);
[~,~,nEdge,nEdgeFeat] = size(edgeMap);
nParamLoc = double(max(nodeMap(:)));
nParamRel = double(max(edgeMap(:))) - nParamLoc;
nStateLoc = nNode * nState;
nStateRel = nEdge * nState^2;

% create local map
F_loc = zeros(nParamLoc,nStateLoc);
for i = 1:nNode
	idx = localIndex(i,1:nState,nState);
	for f = 1:nNodeFeat
		p = double(nodeMap(i,:,f));
		F_loc((idx-1)*nParamLoc+p) = Xnode(1,f,i);
% 		F_loc(sub2ind(size(F_loc),p,idx)) = Xnode(1,f,i);
	end
end

% create relational map
F_rel = zeros(nParamRel,nStateRel);
for e = 1:nEdge
	idx = pairwiseIndex(e,1:nState,1:nState,nNode,nState) - nStateLoc;
	for f = 1:nEdgeFeat
		p = double(edgeMap(:,:,e,f)) - nParamLoc;
		F_rel((idx-1)*nParamRel+p(:)) = Xedge(1,f,e);
% 		F_rel(sub2ind(size(F_rel),p(:),idx)) = Xedge(1,f,e);
	end
end

% combine maps
F = [F_loc sparse(nParamLoc,nEdge*nState^2); sparse(nParamRel,nNode*nState) F_rel];

