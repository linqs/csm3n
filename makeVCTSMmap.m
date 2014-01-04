function F = makeVCTSMmap(Xnode, Xedge, nodeMap, edgeMap)
%
% Creates the feature map for VCTSM learning.
%
% Xnode : nFeat x nNode matrix of observed node features.
% Xedge : nFeat x nEdge matrix of observed edge features.
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
	idx = localIndex(i,1:nState,nNode);
	for f = 1:nNodeFeat
		p = nodeMap(i,:,f);
		F_loc(p,idx) = Xnode(1,f,i);
	end
end

% create relational map
F_rel = zeros(nParamRel,nStateRel);
for e = 1:nEdge
	idx = pairwiseIndex(e,1:nState,1:nState,nNode,nState) - nStateLoc;
	for f = 1:nEdgeFeat
		p = edgeMap(:,:,e,f) - nParamLoc;
		F_rel(p(:),idx) = Xedge(1,f,e);
	end
end

% combine maps
F = [F_loc sparse(nParamLoc,nEdge*nState^2); sparse(nParamRel,nNode*nState) F_rel];

