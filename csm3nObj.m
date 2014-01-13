function [f, g] = csm3nObj(w, examples_l, examples_u, decodeFunc, C, varargin)



% L2 weight regularization
f = 0.5 * (C.*w)' * w;
g = C.*w;

% M3N objective
for i = 1:length(examples_l)
	ex = examples_l{i};
	[l,sg] = UGM_M3N_Obj(w,ex.Xnode,ex.Xedge,ex.Y',ex.nodeMap,ex.edgeMap,ex.edgeStruct,decodeFunc);
	f = f + l;
	g = g + sg;
end

% stability regularization
for i = 1:length(examples_u)
	ex = examples_u{i};
	[stab,sg] = stability(w,ex,decodeFunc,varargin{:});
	f = f + stab;
	g = g + sg;
end


function [stab, sg] = stabilityObj(w, ex, decodeFunc, varargin)
	
	% inferfence for unperturbed input
	[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex.Xnode,ex.Xedge,ex.nodeMap,ex.edgeMap,ex.edgeStruct);
	y_unp = decodeFunc(nodePot,edgePot,ex.edgeStruct,varargin{:});

	% init worst perturbation to current X, and y_per = y_unp
	Xnode = ex.Xnode;
	Xedge = ex.Xedge;
	y_per = y_unp;
	
	% create constraints on perturbation
	
	% find worst perturbation
	
	% NOT NEEDED IF OPTIMIZATION OUTPUTS STABILITY
% 	% loss-augmented inference for perturbed input
% 	Ynode = overcompleteRep(y_unp,ex.nState,0);
% 	y_per = lossAugInfer(w,Xnode,Xedge,Ynode,ex.nodeMap,ex.edgeMap,ex.edgeStruct,decodeFunc,varargin{:});
% 	
% 	% L1 distance between predictions
% 	stab = norm(y_unp-y_per, 1);
	
function [f, g] = perturbObj(x, w, Ynode_unp, nodeMap, edgeMap, edgeStruct, lag, decodeFunc, varargin)
	
	% reconstruct Xnode, Xedge from x
	[nNode,nState,nFeat] = size(nodeMap);
	Xnode = reshape(x, nFeat, nNode);
	Xedge = UGM_makeEdgeFeatures(Xnode,edgeStruct.edgeEnds);
	
	% loss-augmented inference for perturbed input
	y_per = lossAugInfer(w,Xnode,Xedge,Ynode_unp,nodeMap,edgeMap,edgeStruct,decodeFunc,varargin{:});
	Ynode_per = overcompleteRep(y_per,ex.nState,0);

	% L1 distance between predictions
	stab = norm(Ynode_unp - Ynode_per, 1);

	% NOT NEEDED?
% 	% compute objective and gradient
% 	[f1, g1] = Xobj(Wnode,Wedge,Xnode,Ynode_unp,Yedge{1});
% 	[f2, g2] = Xobj(Wnode,Wedge,Xnode,Ynode_per,Yedge{2});
% 	f = lag * (f1 + f2);
% 	g = lag * (g1 + g2);
	
	
	