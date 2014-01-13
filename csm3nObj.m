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

	% find worst perturbation
	Xnode = zeros(1,ex.nFeat,ex.nNode);
	Xedge = zeros(ex.nState,ex.nState,ex.nEdge,ex.nFeat);
	
	% loss-augmented inference for perturbed input
	Ynode = overcompleteRep(y_unp,ex.nState,0);
	y_per = lossAugInfer(w,Xnode,Xedge,Ynode,ex.nodeMap,ex.edgeMap,ex.edgeStruct,decodeFunc,varargin{:});
	
	% L1 distance between predictions
	stab = norm(y_unp-y_per, 1);
	
function [f, g] = perturbObj(x, Wnode, Wedge, Ynode, Yedge, lag)
	
	% reconstruct Xnode from x
	Xnode = zeros(nFeat,nNode);
	
	% compute objective and gradient for Y1,Y2
	[f1, g1] = Xobj(Wnode,Wedge,Xnode,Ynode{1},Yedge{1});
	[f2, g2] = Xobj(Wnode,Wedge,Xnode,Ynode{2},Yedge{2});
	f = lag * (f1 + f2);
	g = lag * (g1 + g2);
	