function [stabMax,stabAvg,perturbs] = measureStabilityRand(params, ex, Xdesc, nSamp, decoder, edgeFeatFunc, y0, perturbs)

% params : cell array of model parameters, where
%			params{1} = w; params{2} = kappa (optional)
% ex : example structure
% Xdesc : X descriptor struct
% nSamp : number of samples
% decoder : decoder function
% edgeFeatFunc : (optional) function to generate edge features (def: UGM_makeEdgeFeatures)
% y0 : (optional) initial predictions
% perturbs: (optional) nFeat+1 x nSamp matrix, where first element in each
%				column is index of perturbed node.
%
% stab : Hamming stability of decoding

% dimensions
[nNode,nState,nFeat] = size(ex.nodeMap);
nEdge = ex.nEdge;

% params
w = params{1};
vc = 0;
if length(params) == 2
	kappa = params{2};
	vc = 1;
end

% X descriptor
if ~isstruct(Xdesc)
	Xdesc = struct('discreteX',1);
end

% edge feature function
if nargin < 6
	edgeFeatFunc = @UGM_makeEdgeFeatures;
end

% unperturbed y
if nargin < 7
	% run initial inference
	if vc == 1
		mu = vctsmInfer(w,kappa,ex.Fx,ex.Aeq,ex.beq);
		y0 = decodeMarginals(mu,nNode,nState);
	else
		[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex.Xnode,ex.Xedge,ex.nodeMap,ex.edgeMap,ex.edgeStruct);
		y0 = decoder(nodePot,edgePot,ex.edgeStruct);
	end
end

% perturbations
if nargin < 8 || isempty(perturbs)
	perturbs = zeros(nFeat+1,nSamp);
	if Xdesc.discreteX
		% select random subset of (node,value) combinations
		otherVals = randsample(find(~(ex.Xnode)),nSamp);
		[featIdx,nodeIdx] = ind2sub([nFeat nNode], otherVals);
		perturbs(1,:) = nodeIdx;
		perturbs(sub2ind(size(perturbs),featIdx+1,(1:nSamp)')) = 1;
	else
		% select iid random (node,feature) combinations
		nodeIdx = randsample(nNode,nSamp,1);
		featIdx = randsample(nFeat,nSamp,1);
		perturbs(1,:) = nodeIdx;
		% uniform random perturbations in [-1,1]
		perturbs(2:end,:) = squeeze(ex.Xnode(1,:,nodeIdx));
		idx = sub2ind([nFeat+1 nSamp],featIdx+1,(1:nSamp)');
		perturbs(idx) = perturbs(idx) + 2*rand(nSamp,1)-1;
		% (might want to project onto L1 ball around original x)
		if isfield(Xdesc,'nonneg') && Xdesc.nonneg
			perturbs(idx) = max(perturbs(idx), zeros(nSamp,1));
		end
	end
else
	nSamp = size(perturbs,2);
end

% random perturbations
stabMax = 0;
stabAvg = 0;
Xnode = ex.Xnode;
for samp = 1:nSamp
	
	% perturb X
	n = perturbs(1,samp);
	x_old = Xnode(1,:,n);
	x_new = perturbs(2:end,samp);
	Xnode(1,:,n) = x_new;
	Xedge = edgeFeatFunc(Xnode,ex.edgeStruct.edgeEnds);

	% run inference
	if vc == 1
		Fx = makeVCTSMmap(Xnode,Xedge,ex.nodeMap,ex.edgeMap);
		mu = vctsmInfer(w,kappa,Fx,ex.Aeq,ex.beq);
		y1 = decodeMarginals(mu,nNode,nState);
	else
		[nodePot,edgePot] = UGM_CRF_makePotentials(w,Xnode,Xedge,ex.nodeMap,ex.edgeMap,ex.edgeStruct);
		y1 = decoder(nodePot,edgePot,ex.edgeStruct);
	end

	% measure Hamming distance of decoding and store max
	delta = nnz(y0 ~= y1);
	if stabMax < delta
		stabMax = delta;
	end
	stabAvg = (1/samp) * delta + ((samp-1)/samp) * stabAvg;
	
	% replace perturbed value
	Xnode(1,:,n) = x_old;
	
end

