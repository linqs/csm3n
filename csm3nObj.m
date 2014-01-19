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
f = 0.5 * (C_w.*w)' * w;
g = C_w.*w;

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
	[stab,sg] = stabilityObj(w,ex,decodeFunc,options,varargin{:});
	f = f + C_s * stab;
	g = g + C_s * sg;
end


	