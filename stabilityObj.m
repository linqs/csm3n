function [f, g] = stabilityObj(w, ex, decodeFunc, varargin)

Xnode = ex.Xnode;
Xedge = ex.Xedge;
nodeMap = ex.nodeMap;
edgeMap = ex.edgeMap;
edgeStruct = ex.edgeStruct;

% inferfence for unperturbed input
[nodePot,edgePot] = UGM_CRF_makePotentials(w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct);
y_unp = decodeFunc(nodePot,edgePot,edgeStruct,varargin{:});
yoc_unp = overcompletePairwise(y_unp,edgeStruct);

% init worst perturbation to current X
x0 = reshape(squeeze(Xnode(1,:,:)),[],1);

% create constraints on perturbation
A = ones(size(x0')) - 2*x0';
b = 1 - sum(x0);
lb = zeros(size(length(x0)));

% perturbation objective
obj = @(x,varargin) perturbObj(x,w,yoc_unp,nodeMap,edgeMap,edgeStruct,decodeFunc,varargin{:});

% find worst perturbation as LP
[x_per,f,~,~,~,g] = fmincon(obj,x0,A,b,[],[],lb);


function [f, g] = perturbObj(x, w, yoc_unp, nodeMap, edgeMap, edgeStruct, decodeFunc, varargin)
	
	% reconstruct Xnode, Xedge from x
	[nNode,nState,nFeat] = size(nodeMap);
	edgeEnds = edgeStruct.edgeEnds;
	Xnode = reshape(x, 1, nFeat, nNode);
	Xedge = UGM_makeEdgeFeatures(Xnode,edgeEnds);
	
	% loss-augmented inference for perturbed input
	Ynode_unp = reshape(yoc_unp(1:(nNode*nState)),nState,nNode);
	y_per = lossAugInfer(w,Xnode,Xedge,Ynode_unp,nodeMap,edgeMap,edgeStruct,decodeFunc,varargin{:});
	yoc_per = overcompletePairwise(y_per,edgeStruct);

	% L1 distance between predictions
	f = norm(yoc_unp - yoc_per, 1);
	
	% objective/gradient w.r.t. x
	Xnode = squeeze(Xnode); % first dimension only has 1 entry
	g = zeros(size(Xnode));
	for i = 1:nNode
		Wnode = squeeze(w(nodeMap(i,:,:)))';
		idx = localIndex(i,1:nState,nState);
		dy = yoc_per(idx) - yoc_unp(idx);
		f = f + sum(sum( Wnode .* (Xnode(:,i) * dy') ));
		g(:,i) = Wnode * dy;
	end
	for e = 1:size(edgeEnds,1)
		Wedge = squeeze(w(edgeMap(:,:,e,:)));
		i = edgeEnds(e,1);
		j = edgeEnds(e,2);
		idx = pairwiseIndex(e,1:nState,1:nState,nNode,nState);
		dy = yoc_per(idx) - yoc_unp(idx);
		f = f + sum(sum( Wedge(:,:,1) .* (Xnode(:,i) * dy') )) ...
			  + sum(sum( Wedge(:,:,2) .* (Xnode(:,j) * dy') ));
		g(:,i) = g(:,i) + Wedge(:,:,1) * dy;
		g(:,j) = g(:,j) + Wedge(:,:,2) * dy;
	end

	% turn the max into a min
	f = -f;
	g = -g;

	
