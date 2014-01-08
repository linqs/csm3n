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


function [f, g] = stabilityObj(x, ex, w, decodeFunc)

	% compute prediction for unperturbed input
	y_unperturb = zeros(10,1);
	
	% compute prediction for perturbed input
	y_perturb = zeros(10,1);
	
	% L1 distance between predictions
	f = norm(y_unperturb-y_perturb, 1);
	