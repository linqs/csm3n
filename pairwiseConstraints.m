function [Aeq, beq] = pairwiseConstraints(edgeStruct)
%
% Generates constraints for a pairwise MRF.
% 
% edgeStruct : edge structure

nNode = double(edgeStruct.nNodes);
nEdge = edgeStruct.nEdges;
nState = double(max(edgeStruct.nStates));
assert(nState == min(edgeStruct.nStates), 'pairwiseConstraints assumes uniform domain for Y')

nCon = nNode + nEdge*(1 + 2*nState);

% fill beq as dense vector
beq = zeros(nCon, 1);
% fill Aeq as sparse matrix
AI = zeros(nNode*nState + nEdge*(3*nState^2 + 2*nState), 1);
AJ = zeros(nNode*nState + nEdge*(3*nState^2 + 2*nState), 1);
AV = zeros(nNode*nState + nEdge*(3*nState^2 + 2*nState), 1);

% iterators
aIter = 1;
bIter = 1;

% local marginal constraints
for i = 1:nNode
    AI(aIter:aIter+nState-1) = bIter;
    AJ(aIter:aIter+nState-1) = localIndex(i,1:nState,nState);
    AV(aIter:aIter+nState-1) = 1;
    beq(bIter) = 1;
	aIter = aIter + nState;
    bIter = bIter + 1;
end

% pairwise (pseudo)marginal constraints
for e = 1:nEdge
	
	% node indices
	n1 = edgeStruct.edgeEnds(e,1);
	n2 = edgeStruct.edgeEnds(e,2);
	
    % make marginal sum to 1    
    AI(aIter:aIter+nState^2-1) = bIter;
    AJ(aIter:aIter+nState^2-1) = pairwiseIndex(e,1:nState,1:nState,nNode,nState);
    AV(aIter:aIter+nState^2-1) = 1;
    beq(bIter) = 1;
	aIter = aIter + nState^2;
    bIter = bIter + 1;
    
    for s = 1:nState
        % marginalize over a
        AI(aIter:aIter+nState-1) = bIter;
        AJ(aIter:aIter+nState-1) = pairwiseIndex(e,1:nState,s,nNode,nState);
        AV(aIter:aIter+nState-1) = 1;

        AI(aIter+nState) = bIter;
        AJ(aIter+nState) = localIndex(n2,s,nState);
        AV(aIter+nState) = -1;
        beq(bIter) = 0;
        
		aIter = aIter + nState + 1;
        bIter = bIter + 1;
        
        % marginalize over b
        AI(aIter:aIter+nState-1) = bIter;
        AJ(aIter:aIter+nState-1) = pairwiseIndex(e,s,1:nState,nNode,nState);
        AV(aIter:aIter+nState-1) = 1;

        AI(aIter+nState) = bIter;
        AJ(aIter+nState) = localIndex(n1,s,nState);
        AV(aIter+nState) = -1;
        beq(bIter) = 0;
        
		aIter = aIter + nState + 1;
        bIter = bIter + 1;
    end
    
end

% create sparse Aeq using indices AI,AJ, values AV
Aeq = sparse(AI, AJ, AV, nCon, nNode*nState + nEdge*nState^2);



