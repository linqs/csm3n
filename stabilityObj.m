function [f, sg, x_p, y_p] = stabilityObj(w, ex, decodeFunc, varargin)
% 
% Computes the stability regularization objective and gradient.
% 
% w : nParam x 1 vector of weights
% ex : an unlabeled example
% decodeFunc : decoder function

nodeMap = ex.nodeMap;
edgeMap = ex.edgeMap;
edgeStruct = ex.edgeStruct;
edgeEnds = edgeStruct.edgeEnds;
[nNode,nState,nFeat] = size(nodeMap);

%% INFER UNPERTURBED EXAMPLE

% inferfence for unperturbed input
Xnode_u = ex.Xnode;
Xedge_u = ex.Xedge;
x_u = reshape(squeeze(Xnode_u(1,:,:)),[],1);
[nodePot,edgePot] = UGM_CRF_makePotentials(w,Xnode_u,Xedge_u,nodeMap,edgeMap,edgeStruct);
y_u = decodeFunc(nodePot,edgePot,edgeStruct,varargin{:});
yoc_u = overcompletePairwise(y_u,edgeStruct);
Ynode_u = reshape(yoc_u(1:(nNode*nState)),nState,nNode);

%% FIND WORST PERTURBATION

% init perturbation to current x, with buffer
x0 = min(max(x_u,.00001),.99999);

% create constraints on perturbation
A = ones(size(x0')) - 2*x0';
b = 1 - sum(x0);
lb = zeros(size(x0));
ub = ones(size(x0));

% perturbation objective
objFun = @(x,varargin) perturbObj(x,w,yoc_u,Ynode_u,nodeMap,edgeMap,edgeStruct,decodeFunc,varargin{:});

% projection function
projFun = @(x) perturbProj(x,x_u);

% % find worst perturbation using IPM
% options = optimset('GradObj','on','Algorithm','interior-point' ...
% 				  ,'Display','iter','MaxIter',100 ...
% 				  ,'TolFun',1e-3,'TolX',1e-8,'TolCon',1e-5);
% [x_p,f] = fmincon(objFun,x0,A,b,[],[],lb,ub,[],options);

% find worst perturbation using PGD
options.maxIter = 10;
options.stepSize = 1e-3;
options.fTol = 1e-4;
% options.verbose = 1;
[x_p,f] = pgd(objFun,projFun,x0,options);

% convert min to max
f = -f;


%% GRADIENT w.r.t. WEIGHTS

% reconstruct Xnode,Xedge from x_p
Xnode_p = reshape(x_p, 1, nFeat, nNode);
Xedge_p = UGM_makeEdgeFeatures(Xnode_p,edgeEnds);

% loss-augmented inference for perturbed input
y_p = lossAugInfer(w,Xnode_p,Xedge_p,Ynode_u,nodeMap,edgeMap,edgeStruct,decodeFunc,varargin{:});
yoc_p = overcompletePairwise(y_p,edgeStruct);

% only 1 example
Xnode_p = squeeze(Xnode_p);
Xedge_p = squeeze(Xedge_p);

% compute (sub)gradient w.r.t. w
sg = zeros(size(w));
widx = squeeze(nodeMap(1,:,:))';
idx = localIndex(1,1:nState,nState);
for i = 1:nNode
	dy = yoc_p(idx) - yoc_u(idx);
	sg(widx) = sg(widx) + Xnode_p(:,i) * dy';
	idx = idx + nState;
end
widx = reshape(squeeze(edgeMap(:,:,1,:)),nState^2,2*nFeat)';
idx = pairwiseIndex(1,1:nState,1:nState,nNode,nState);
for e = 1:size(edgeEnds,1)
	i = edgeEnds(e,1);
	j = edgeEnds(e,2);
	dy = yoc_p(idx) - yoc_u(idx);
	sg(widx) = sg(widx) + Xedge_p(:,e) * dy';
	idx = idx + nState^2;
end


%% LOG

% worst perturbation
[mxv,mxi] = max(abs(x_u-x_p));

% fprintf('Worst perturbation: (%d, %f)\n', mxi,mxv);
% fprintf('Perturbation objective: %f\n', f);
% fprintf('Stability (L1-distance): %f\n', norm(yoc_u-yoc_p,1));


%% PERTURBATION OBJECTIVE

function [f, g] = perturbObj(x, w, yoc_u, Ynode_u, nodeMap, edgeMap, edgeStruct, decodeFunc, varargin)

	% reconstruct Xnode,Xedge from x
	[nNode,nState,nFeat] = size(nodeMap);
	edgeEnds = edgeStruct.edgeEnds;
	Xnode_p = reshape(x, 1, nFeat, nNode);
	Xedge_p = UGM_makeEdgeFeatures(Xnode_p,edgeEnds);
	
	% loss-augmented inference for perturbed input
	y_p = lossAugInfer(w,Xnode_p,Xedge_p,Ynode_u,nodeMap,edgeMap,edgeStruct,decodeFunc,varargin{:});
	yoc_p = overcompletePairwise(y_p,edgeStruct);

	% L1 distance between predictions
	stab = norm(yoc_u - yoc_p, 1);
	
	% only 1 example
	Xnode_p = squeeze(Xnode_p);
	Xedge_p = squeeze(Xedge_p);
	
	% objective/gradient w.r.t. x
	f = 0;
	g = zeros(size(Xnode_p));
	Wnode = squeeze(w(nodeMap(1,:,:)))';
	idx = localIndex(1,1:nState,nState);
	for i = 1:nNode
		dy = yoc_p(idx) - yoc_u(idx);
		f = f + sum(sum( Wnode .* (Xnode_p(:,i) * dy') ));
		g(:,i) = Wnode * dy;
		idx = idx + nState;
	end
	Wedge = w(reshape(squeeze(edgeMap(:,:,1,:)),nState^2,2*nFeat)');
	idx = pairwiseIndex(1,1:nState,1:nState,nNode,nState);
	for e = 1:size(edgeEnds,1)
		i = edgeEnds(e,1);
		j = edgeEnds(e,2);
		dy = yoc_p(idx) - yoc_u(idx);
		f = f + sum(sum( Wedge .* (Xedge_p(:,e) * dy') ));
		g(:,i) = g(:,i) + Wedge(1:nFeat,:) * dy;
		g(:,j) = g(:,j) + Wedge(nFeat+1:2*nFeat,:) * dy;
		idx = idx + nState^2;
	end

	% convert max to min
	f = -(f + stab);
	g = -g(:);

%% PERTURBATION PROJECTION

function x_p = perturbProj(v, x_u)

	x_p = projectOntoL1Ball(v - x_u, 1);
	x_p = x_p + x_u;

	
