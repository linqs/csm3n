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
nParamLoc = max(nodeMap(:));
nParamRel = max(edgeMap(:));

% create local map
F_loc = zeros(nParamLoc,nNode*nState);
for i = 1:nNode
	for s = 1:nState
		for f = 1:nNodeFeat
			p = nodeMap(i,s,f);
			F_loc(p,(i-1)*nState+s) = Xnode(f,i);
		end
	end
end

% create relational map
F_rel = zeros(nParamRel,nEdge*nState^2);
for s1 = 1:nState
	for s2 = 1:nState
		for e = 1:nEdge
			for f = 1:nEdgeFeat
				p = edgeMap(s1,s2,e,f);
				F_rel(p,(i-1)*nState^2+(s1-1)*nState+s2) = Xedge(f,e);
			end
		end
	end
end

% combine maps
F = [F_loc sparse(size(F_rel)); sparse(size(F_loc)) F_rel];

