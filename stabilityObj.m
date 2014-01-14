function [f, g, x_per] = stabilityObj(w, ex, decodeFunc, varargin)

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
lb = zeros(size(x0));
ub = ones(size(x0));

% perturbation objective
obj = @(x,varargin) perturbObj(x,w,yoc_unp,nodeMap,edgeMap,edgeStruct,decodeFunc,varargin{:});

% find worst perturbation as LP
x0 = min(max(x0,.00001),.99999);
options = optimset('GradObj','on','Algorithm','interior-point' ...
				  ,'Display','iter','MaxIter',100 ...
				  ,'TolFun',1e-3,'TolX',1e-8,'TolCon',1e-5);
[x_per,f,~,~,~,g] = fmincon(obj,x0,A,b,[],[],lb,ub,[],options);
f = -f;
g = -g;


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
	stab = norm(yoc_unp - yoc_per, 1);
	
	% objective/gradient w.r.t. x
	Xnode = squeeze(Xnode); % first dimension only has 1 entry
	f = 0;
	g = zeros(size(Xnode));
	Wnode = squeeze(w(nodeMap(1,:,:)))';
	idx = localIndex(1,1:nState,nState);
	for i = 1:nNode
		dy = yoc_per(idx) - yoc_unp(idx);
		f = f + sum(sum( Wnode .* (Xnode(:,i) * dy') ));
		g(:,i) = Wnode * dy;
		idx = idx + nState;
	end
	Wedge = w(reshape(squeeze(edgeMap(:,:,1,:)),nState^2,2*nFeat)');
	idx = pairwiseIndex(1,1:nState,1:nState,nNode,nState);
	for e = 1:size(edgeEnds,1)
		i = edgeEnds(e,1);
		j = edgeEnds(e,2);
		dy = yoc_per(idx) - yoc_unp(idx);
		f = f + sum(sum( Wedge .* [(Xnode(:,i) * dy') ; (Xnode(:,j) * dy')] ));
		g(:,i) = g(:,i) + Wedge(1:nFeat,:) * dy;
		g(:,j) = g(:,j) + Wedge(nFeat+1:2*nFeat,:) * dy;
		idx = idx + nState^2;
	end

	% turn the max into a min
	f = -(f + stab);
	g = -g(:);

	
