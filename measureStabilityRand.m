function [stabMax,stabAvg,perturbs] = measureStabilityRand(params, ex, discreteX, nSamp, decoder, y0, perturbs)

% params : cell array of model parameters, where
%			params{1} = w; params{2} = kappa (optional)
% ex : example structure
% discreteX : whether X is discrete
% nSamp : number of samples
% decoder : decoder function
% y0 : (optional) initial predictions
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

% unperturbed y
if nargin < 6
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
if nargin < 7 || isempty(perturbs)
	% select random subset of (node,value) combinations
	if discreteX
		otherVals = randsample(find(~ex.Xnode),nSamp);
		[I,J] = ind2sub([nFeat nNode], otherVals);
		perturbs = zeros(nFeat+1,nSamp);
		perturbs(1,:) = J;
		perturbs(sub2ind(size(perturbs),I+1,(1:nSamp)')) = 1;
	else

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
	Xedge = UGM_makeEdgeFeatures(Xnode,ex.edgeStruct.edgeEnds);

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

