function [f, g] = csm3nObj(w, examples_l, examples_u, decodeFunc, C_w, C_s, options, varargin)
% 
% Computes the CSM3N objective and gradient.
% 
% w : nParam x 1 vector of weights
% examples_l : cell array of labeled examples, used for M3N training
% examples_u : cell array of unlabeled examples, used for stability regularization
% decodeFunc : decoder function
% C_w : weight regularization constant or nParam x 1 vector
% C_s : stability regularization constant
% options : optional struct of optimization options for stabilityObj

if nargin < 7
	options = struct();
end

% L2 weight regularization
f = 0.5 * (C_w .* w)' * w;
g = C_w .* w;

% M3N objective
nEx_l = length(examples_l);
for i = 1:nEx_l
	ex = examples_l{i};
	[l,sg] = UGM_M3N_Obj(w,ex.Xnode,ex.Xedge,ex.Y',ex.nodeMap,ex.edgeMap,ex.edgeStruct,decodeFunc,varargin{:});
	f = f + l / nEx_l;
	g = g + sg / nEx_l;
end

% stability regularization
nEx_u = length(examples_u);
for i = 1:nEx_u
	ex = examples_u{i};
	[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex.Xnode,ex.Xedge,ex.nodeMap,ex.edgeMap,ex.edgeStruct);
	y = decodeFunc(nodePot,edgePot,ex.edgeStruct,varargin{:});
	[stab,sg] = stabilityObj(w,ex,y,decodeFunc,options,varargin{:});
	f = f + C_s * stab / nEx_u;
	g = g + C_s * sg / nEx_u;
end


	