function [f, g] = csm3nObj(w, examples_l, examples_u, decodeFunc, C, varargin)



loss = 0;
g = 0;
for i = 1:length(examples_l)
	% compute M3N objective
	[l,sg] = UGM_M3N_Obj(w,ex.Xnode,ex.Xedge,ex.Y',ex.nodeMap,ex.edgeMap,ex.edgeStruct,decodeFunc);
	loss = loss + l;
	g = g + sg;
end

% L2 weight regularization
f = loss + 0.5 * (C.*w)' * w;
g = g + C.*w;

% stability regularization


function [f, g] = stabilityObj(x, ex, w)

	% find worst perturbation
	
	% loss-augmented inferfence for unperturbed input
	y_unperturb = zeros(10,1);
	
	% loss-augmented inference for perturbed input
	y_perturb = zeros(10,1);
	
	% L1 distance between predictions
	f = norm(y_unperturb-y_perturb, 1);
	
function [f, g] = perturbObj(x, Wnode, Wedge, Ynode, Yedge, lag)
	
	% reconstruct Xnode from x
	Xnode = zeros(nFeat,nNode);
	
	% compute objective and gradient for Y1,Y2
	[f1, g1] = Xobj(Wnode,Wedge,Xnode,Ynode{1},Yedge{1});
	[f2, g2] = Xobj(Wnode,Wedge,Xnode,Ynode{2},Yedge{2});
	f = lag * (f1 + f2);
	g = lag * (g1 + g2);
	