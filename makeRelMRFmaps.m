function [nodeMap,edgeMap,w] = makeRelMRFmaps(edgeStruct,edgeType)
%
% Creates the node/edge maps and init weights for Mark Schmidt's code,
% where local potentials (X,Y) have different parameters from relational
% potentials (Y,Y).
%
% NOTE: Assumes same number of states for all nodes.
%		UGM indexes params ROW-WISE, e.g., [1 2; 3 4];
%		This code indexes COLUMN-WISE, e.g., [1 3; 2 4].
%
% edgeStruct : UGM edge structure
% edgeType : nEdge x 1 vector of edge types
%              e.g., 1=(Y,Y); 2=(X1,Y); 3=(X2,Y); etc.

assert(min(edgeStruct.nStates)==max(edgeStruct.nStates),...
		'makeRelMRFmaps only supports uniform domains');

nNode = edgeStruct.nNodes;
nEdge = edgeStruct.nEdges;
nState = edgeStruct.nStates(1);
nType = max(edgeType);
nParamNode = nState - 1;
nParamEdge = nType * nState^2;
nParam = nParamNode + nParamEdge;

nodeMap = zeros(nNode,nState);
edgeMap = zeros(nState,nState,nEdge);

for n = 1:nNode
	for s = 2:nState
		nodeMap(n,s) = s-1;
	end
end

for e = 1:nEdge
	t = edgeType(e);
	for s1 = 1:nState
		for s2 = 1:nState
			edgeMap(s1,s2,e) = nParamNode + (t-1)*nState^2 + (s2-1)*nState + s1;
		end
	end
end

w = zeros(nParam,1);
