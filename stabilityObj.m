function [f, sg, x_p, y_p] = stabilityObj(w, ex, y_u, decodeFunc, options, varargin)
% 
% Computes the stability regularization objective and gradient.
% 
% w : nParam x 1 vector of weights
% ex : an (unlabeled) example
% y_u : predicted (or true) label for ex
% decodeFunc : decoder function
% options : optional struct of options:
% 			maxIter : iterations of PGD (def: 10)
% 			stepSize : PGD step size (def: 1e-3)
% 			verbose : verbose mode (def: 0)
% 			plotObj : plot stability objective (def: 0)

if nargin < 5 || ~isstruct(options)
	options = struct();
end
if ~isfield(options,'maxIter')
	options.maxIter = 10;
end
if ~isfield(options,'stepSize')
	options.stepSize = 1e-3;
end
if ~isfield(options,'verbose')
	options.verbose = 0;
end
if ~isfield(options,'plotObj')
	options.plotObj = 0;
end

nodeMap = ex.nodeMap;
edgeMap = ex.edgeMap;
edgeStruct = ex.edgeStruct;
edgeEnds = edgeStruct.edgeEnds;
[nNode,nState,nFeat] = size(nodeMap);
nEdge = size(edgeEnds,1);

x_u = ex.Xnode(:);
yoc_u = overcompletePairwise(y_u,edgeStruct);
Ynode_u = reshape(yoc_u(1:(nNode*nState)),nState,nNode);

%% FIND WORST PERTURBATION

% init perturbation to current x, with buffer
x0 = x_u;%min(max(x_u,.00001),.99999);

% perturbation objective
objFun = @(x,varargin) perturbObj(x,w,yoc_u,Ynode_u,nodeMap,edgeMap,edgeStruct,decodeFunc,varargin{:});

% projection function
projFun = @(x) perturbProj(x,x_u);

% % check gradient calculations
% fastDerivativeCheck(objFun,x0);
% return;

% % find worst perturbation using IPM
% A = ones(size(x0')) - 2*x0';
% b = 1 - sum(x0);
% lb = zeros(size(x0));
% ub = ones(size(x0));
% options = optimset('GradObj','on','Algorithm','interior-point' ...
% 				  ,'Display','iter','MaxIter',100 ...
% 				  ,'TolFun',1e-3,'TolX',1e-8,'TolCon',1e-5);
% [x_p,f] = fmincon(objFun,x0,A,b,[],[],lb,ub,[],options);

% find worst perturbation using PGD
[x_p,f,fVec] = pgd(objFun,projFun,x0,options);

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
Xnode_p = reshape(Xnode_p,nFeat,nNode);
Xedge_p = reshape(Xedge_p,2*nFeat,nEdge);

% compute (sub)gradient w.r.t. w
sg = zeros(size(w));
widx = reshape(nodeMap(1,:,:),nState,nFeat);
yidx = localIndex(1,1:nState,nState);
for i = 1:nNode
	dy = yoc_p(yidx) - yoc_u(yidx);
	sg(widx) = sg(widx) + dy * Xnode_p(:,i)';
	yidx = yidx + nState;
end
widx = reshape(edgeMap(:,:,1,:),nState^2,2*nFeat);
yidx = pairwiseIndex(1,1:nState,1:nState,nNode,nState);
for e = 1:size(edgeEnds,1)
	dy = yoc_p(yidx) - yoc_u(yidx);
	sg(widx) = sg(widx) + dy * Xedge_p(:,e)';
	yidx = yidx + nState^2;
end


%% LOG

if options.verbose
	[mxv,mxi] = max(abs(x_u-x_p));
	fprintf('Worst perturbation: (%d, %f)\n', mxi,mxv);
	fprintf('Perturbation objective: %f\n', f);
	fprintf('Stability (L1-distance): %f\n', norm(yoc_u-yoc_p,1));
end

if options.plotObj
	plot(1:length(fVec),fVec);
end


%% PERTURBATION OBJECTIVE

function [f, g] = perturbObj(x, w, yoc_u, Ynode_u, nodeMap, edgeMap, edgeStruct, decodeFunc, varargin)

	edgeEnds = edgeStruct.edgeEnds;
	[nNode,nState,nFeat] = size(nodeMap);
	nEdge = size(edgeEnds,1);

	% reconstruct Xnode,Xedge from x
	Xnode_p = reshape(x, 1, nFeat, nNode);
	Xedge_p = UGM_makeEdgeFeatures(Xnode_p,edgeEnds);
	
	% loss-augmented inference for perturbed input
	y_p = lossAugInfer(w,Xnode_p,Xedge_p,Ynode_u,nodeMap,edgeMap,edgeStruct,decodeFunc,varargin{:});
	yoc_p = overcompletePairwise(y_p,edgeStruct);

	% L1 distance between predictions
	stab = norm(yoc_u - yoc_p, 1);
	
	% only 1 example
	Xnode_p = reshape(Xnode_p,nFeat,nNode);
	Xedge_p = reshape(Xedge_p,2*nFeat,nEdge);
	
	% objective/gradient w.r.t. x
	f = 0;
	g = zeros(size(Xnode_p));
	Wnode = reshape(w(nodeMap(1,:,:)),nState,nFeat)';
	yidx = localIndex(1,1:nState,nState);
	for i = 1:nNode
		dy = yoc_p(yidx) - yoc_u(yidx);
		f = f + sum(sum( Wnode .* (Xnode_p(:,i) * dy') ));
		g(:,i) = Wnode * dy;
		yidx = yidx + nState;
	end
	Wedge = w(reshape(edgeMap(:,:,1,:),nState^2,2*nFeat)');
	yidx = pairwiseIndex(1,1:nState,1:nState,nNode,nState);
	for e = 1:size(edgeEnds,1)
		i = edgeEnds(e,1);
		j = edgeEnds(e,2);
		dy = yoc_p(yidx) - yoc_u(yidx);
		f = f + sum(sum( Wedge .* (Xedge_p(:,e) * dy') ));
		g(:,i) = g(:,i) + Wedge(1:nFeat,:) * dy;
		g(:,j) = g(:,j) + Wedge(nFeat+1:2*nFeat,:) * dy;
		yidx = yidx + nState^2;
	end

	% convert max to min
	f = -(f + stab);
	g = -g(:);

%% PERTURBATION PROJECTION

function x_p = perturbProj(v, x_u)

	x_p = projectOntoL1Ball(v - x_u, 1);
	x_p = x_p + x_u;

	
