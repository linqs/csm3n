function examples = loadCora()

load cora.mat;

% remove diagonal from G
G = G - diag(diag(G));

% partition into 4 networks
nEx = 4;
nNode = length(y);
nNodeEx = floor(nNode / nEx);
nState = 6;

for i = 1:nEx
	
	idx = (i-1)*nNodeEx+1:i*nNodeEx;
	
	edgeStruct = UGM_makeEdgeStruct(G(idx,idx),nState,1);
% 	subplot(2,2,i);
% 	spy(G(idx,idx));
	
	[Aeq,beq] = pairwiseConstraints(edgeStruct);
	
	Xnode(1,:,:) = full(X(idx,:)');
	Xedge = makeEdgeFeatures(Xnode,edgeStruct.edgeEnds);
	
	examples{i} = makeExample(Xnode,Xedge,y(idx),nState,edgeStruct,Aeq,beq);

end


