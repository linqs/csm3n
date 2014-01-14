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
	[stab,sg] = stabilityObj(w,ex,decodeFunc,varargin{:});
	f = f + stab;
	g = g + sg;
end


	