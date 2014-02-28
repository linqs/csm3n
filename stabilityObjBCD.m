function [f, sg, x_p, y_p] = stabilityObjBCD(w, ex, y_u, decodeFunc, options, varargin)
% 
% Computes the stability regularization objective and gradient,
%  using block coordinate descent.
% 
% w : nParam x 1 vector of weights
% ex : an (unlabeled) example
% y_u : predicted (or true) label for ex
% decodeFunc : decoder function
% options : optional struct of options:
%			edgeFeatFunc : function to generate edge features
% 			maxIter : iterations of PGD (def: 10)
% 			stepSize : PGD step size (def: 1e-3)
% 			verbose : verbose mode (def: 0)
% 			plotObj : plot stability objective (def: 0)

if nargin < 5 || ~isstruct(options)
	options = struct();
end
if ~isfield(options,'edgeFeatFunc')
	options.edgeFeatFunc = @makeEdgeFeatures;
end
if ~isfield(options,'maxIter')
	options.maxIter = 10;
end
if ~isfield(options,'stepSize')
	options.stepSize = .1;
end
if ~isfield(options,'verbose')
	options.verbose = 0;
end
if ~isfield(options,'plotObj')
	options.plotObj = 0;
end

nNode = ex.nNode;
nEdge = ex.nEdge;
nState = ex.nState;
nNodeFeat = ex.nNodeFeat;
nEdgeFeat = ex.nEdgeFeat;
nodeMap = ex.nodeMap;
edgeMap = ex.edgeMap;
edgeStruct = ex.edgeStruct;
edgeEnds = edgeStruct.edgeEnds;

Wnode = reshape(w(nodeMap(1,:,:)),nState,nNodeFeat)';
Wedge = w(reshape(edgeMap(:,:,1,:),nState^2,nEdgeFeat)');

x_u = ex.Xnode(:);
yoc_u = overcompletePairwise(y_u,edgeStruct);
Ynode_u = reshape(yoc_u(1:(nNode*nState)),nState,nNode);

%% BLOCK COORDINATE DESCENT

fVec = [];

% init perturbation to current x, with buffer
x_p = x_u;%min(max(x_u,.00001),.99999);
Xnode_p = ex.Xnode;
Xedge_p = ex.Xedge;

for t = 1:options.maxIter
	
	% loss-augmented inference for perturbed input
	edgeStructCopy = edgeStruct;
	edgeStructCopy.maxIter = 10;
	y_p = lossAugInfer(w,Xnode_p,Xedge_p,Ynode_u,nodeMap,edgeMap,edgeStructCopy,decodeFunc,varargin{:});
	yoc_p = overcompletePairwise(y_p,edgeStruct);
% 	yoc_p = yoc_p + (options.stepSize / sqrt(t)) * (yoc_p_new - yoc_p);
	
	% solve for worst perturbation as LP
	F = zeros(nNodeFeat,nNode);
	yidx = localIndex(1,1:nState,nState);
	for i = 1:nNode
		dy = yoc_p(yidx) - yoc_u(yidx);
		F(:,i) = Wnode * dy;
		yidx = yidx + nState;
	end
	yidx = pairwiseIndex(1,1:nState,1:nState,nNode,nState);
	for e = 1:size(edgeEnds,1)
		i = edgeEnds(e,1);
		j = edgeEnds(e,2);
		dy = yoc_p(yidx) - yoc_u(yidx);
		% following lines assume that edge-specific features occur
		% after concatenation of edge features.
		F(:,i) = F(:,i) + Wedge(1:nNodeFeat,:) * dy;
		F(:,j) = F(:,j) + Wedge(nNodeFeat+1:2*nNodeFeat,:) * dy;
		yidx = yidx + nState^2;
	end
	A = ones(size(x_p')) - 2*x_u';
	b = 1 - sum(x_u);
	lb = zeros(size(x_p));
	ub = ones(size(x_p));
	optLP = optimset('Display','off','MaxIter',10);
% 						,'GradObj','on','Algorithm','interior-point' ...
% 						,'TolFun',1e-3,'TolX',1e-8,'TolCon',1e-5);
	x_p_new = linprog(-F(:),A,b,[],[],lb,ub,[],optLP);
	x_p = x_p + (options.stepSize / sqrt(t)) * (x_p_new - x_p);
	
	% project perturbation back into L1 ball around unperturbed
% 	x_p = x_u + projectOntoL1Ball(x_p - x_u, 1);
	
	% reconstruct Xnode,Xedge from x_p
	Xnode_p = reshape(x_p, 1, nNodeFeat, nNode);
	Xedge_p = options.edgeFeatFunc(Xnode_p,edgeEnds);
	
	% perturbation objective
	[f,stab] = perturbObj(Wnode, Wedge, yoc_u, yoc_p, Xnode_p, Xedge_p, nNode, nEdge, nState);
	
	fprintf('Perturb obj = %f; Stabilty = %f\n',f,stab);

end


%% GRADIENT w.r.t. WEIGHTS

% loss-augmented inference for perturbed input
y_p = lossAugInfer(w,Xnode_p,Xedge_p,Ynode_u,nodeMap,edgeMap,edgeStruct,decodeFunc,varargin{:});
yoc_p = overcompletePairwise(y_p,edgeStruct);

% perturbation objective
[f,stab] = perturbObj(Wnode, Wedge, yoc_u, yoc_p, Xnode_p, Xedge_p, nNode, nEdge, nState);

% compute (sub)gradient w.r.t. w
sg = zeros(size(w));
widx = reshape(nodeMap(1,:,:),nState,nNodeFeat);
yidx = localIndex(1,1:nState,nState);
for i = 1:nNode
	dy = yoc_p(yidx) - yoc_u(yidx);
	sg(widx) = sg(widx) + dy * Xnode_p(1,:,i);
	yidx = yidx + nState;
end
widx = reshape(edgeMap(:,:,1,:),nState^2,nEdgeFeat);
yidx = pairwiseIndex(1,1:nState,1:nState,nNode,nState);
for e = 1:size(edgeEnds,1)
	dy = yoc_p(yidx) - yoc_u(yidx);
	sg(widx) = sg(widx) + dy * Xedge_p(1,:,e);
	yidx = yidx + nState^2;
end


%% DISPLAY

if options.verbose
	[mxv,mxi] = max(abs(x_u-x_p));
	fprintf('Worst perturbation: (%d, %f)\n', mxi,mxv);
	fprintf('Perturbation objective: %f\n', f);
	fprintf('Stability (L1-distance): %f\n', stab);
end

if options.plotObj
	plot(1:length(fVec),fVec);
end


%% PERTURBATION OBJECTIVE

function [f,stab] = perturbObj(Wnode, Wedge, yoc_u, yoc_p, Xnode_p, Xedge_p, nNode, nEdge, nState)
	
	% L1 distance
	stab = norm(yoc_u - yoc_p, 1);
	f = stab;
	
	% difference of energies
	yidx = localIndex(1,1:nState,nState);
	for i = 1:nNode
		dy = yoc_p(yidx) - yoc_u(yidx);
		f = f + sum(sum( Wnode .* (Xnode_p(1,:,i)' * dy') ));
		yidx = yidx + nState;
	end
	yidx = pairwiseIndex(1,1:nState,1:nState,nNode,nState);
	for e = 1:nEdge
		dy = yoc_p(yidx) - yoc_u(yidx);
		f = f + sum(sum( Wedge .* (Xedge_p(1,:,e)' * dy') ));
		yidx = yidx + nState^2;
	end

	
