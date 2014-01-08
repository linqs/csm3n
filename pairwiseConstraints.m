function [Aeq, beq] = pairwiseConstraints(edgeStruct)
%
% Generates constraints for a pairwise MRF.
% 
% edgeStruct : edge structure

nNode = double(edgeStruct.nNodes);
nEdge = edgeStruct.nEdges;
nState = double(max(edgeStruct.nStates));
assert(nState == min(edgeStruct.nStates), 'pairwiseConstraints assumes uniform domain for Y')

% fill beq as dense vector
beq = zeros(nNode + nEdge*(1 + 2*nState), 1);
% fill Aeq as sparse matrix
AI = [];
AJ = [];
AV = [];

% constraint iterator
iter = 1;

% local marginal constraints
for i = 1:nNode
    AI(end+1:end+nState) = iter;
    AJ(end+1:end+nState) = localIndex(i,1:nState,nState);
    AV(end+1:end+nState) = 1;
    beq(iter) = 1;
    iter = iter + 1;
end

% pairwise (pseudo)marginal constraints
for e = 1:nEdge
	
	% node indices
	n1 = edgeStruct.edgeEnds(e,1);
	n2 = edgeStruct.edgeEnds(e,2);
	
    % make marginal sum to 1    
    AI(end+1:end+nState^2) = iter;
    AJ(end+1:end+nState^2) = pairwiseIndex(e,1:nState,1:nState,nNode,nState);
    AV(end+1:end+nState^2) = 1;
    beq(iter) = 1;
    iter = iter + 1;
    
    for s = 1:nState
        % marginalize over a
        AI(end+1:end+nState) = iter;
        AJ(end+1:end+nState) = pairwiseIndex(e,1:nState,s,nNode,nState);
        AV(end+1:end+nState) = 1;

        AI(end+1) = iter;
        AJ(end+1) = localIndex(n2,s,nState);
        AV(end+1) = -1;
        beq(iter) = 0;
        
        iter = iter + 1;
        
        % marginalize over b
        AI(end+1:end+nState) = iter;
        AJ(end+1:end+nState) = pairwiseIndex(e,s,1:nState,nNode,nState);
        AV(end+1:end+nState) = 1;

        AI(end+1) = iter;
        AJ(end+1) = localIndex(n1,s,nState);
        AV(end+1) = -1;
        beq(iter) = 0;
        
        iter = iter + 1;
    end
    
end

% create sparse Aeq using indices AI,AJ, values AV
Aeq = sparse(AI, AJ, AV, nNode + nEdge*(1 + 2*nState), nNode*nState + nEdge*nState^2);



