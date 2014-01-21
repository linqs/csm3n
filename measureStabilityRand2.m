function [stabMax,stabAvg] = measureStabilityRand2(params, ex, discreteX, nSamp, decoder, y0)

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

% select random subset of (node,value) combinations
if discreteX
	otherVals = randsample(find(~ex.Xnode),nSamp);
	[I,J] = ind2sub([nFeat nNode], otherVals);
else
	
end

% random perturbations
stabMax = 0;
stabAvg = 0;
Xnode = ex.Xnode;
for samp=1:nSamp
	
	% perturb X
	n = J(samp);
	x_old = Xnode(1,:,n);
	if discreteX
		s = I(samp);
		x_new = zeros(size(x_old));
		x_new(s) = 1;
	else
		
	end
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

