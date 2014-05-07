function Fx = makeFx(Xnode, Xedge, nodeMap, edgeMap)
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

% Create local map
F_loc = zeros(nParamLoc,nState,nNode);
for n = 1:nNode
	for s = 1:nState
		for f = 1:nNodeFeat
			p = double(nodeMap(n,s,f));
			F_loc(p,s,n) = Xnode(1,f,n);
		end
	end
end

% Create relational map
F_rel = zeros(nParamRel,nState,nState,nEdge);
for e = 1:nEdge
	for s1 = 1:nState
		for s2 = 1:nState
			for f = 1:nEdgeFeat
				p = double(edgeMap(s1,s2,e,f)) - nParamLoc;
				F_rel(p,s1,s2,e) = Xedge(1,f,e);
			end
		end
	end
end

% Combine maps
Fx = zeros(nParamLoc+nParamRel,nStateLoc+nStateRel);
Fx(1:nParamLoc,1:nStateLoc) = F_loc(:,:);
Fx(nParamLoc+1:end,nStateLoc+1:end) = F_rel(:,:);

