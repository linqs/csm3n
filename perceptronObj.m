function [f, g] = perceptronObj(x, examples, C, decodeFunc, varargin)
%
% Outputs the objective value and gradient of the M3N learning objective
%
% x : current point in optimization
% examples : nEx x 1 cell array of examples, each containing:
%	Fx : nParam x length(oc) feature map
%	suffStat : nParam x 1 vector of sufficient statistics (i.e., Fx * oc)
%	Ynode : nState x nNode overcomplete matrix representation of labels
% C : regularization constant or vector
% decodeFunc : decoder function
% varargin : optional arguments

nEx = length(examples);
nParam = max(examples{1}.edgeMap(:));

% Parse current position
w = x(1:nParam);

% Init outputs
f = 0.5 * (C .* w)' * w;
if nargout == 2
	gradW = (C .* w);
end

% Main loop
for i = 1:nEx
	
	% Grab ith example.
	ex = examples{i};
	Fx = ex.Fx;
	ss_y = ex.suffStat;
	Ynode = ex.Ynode; % assumes Ynode is (nState x nNode)
	
	% Inference
	[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex.Xnode,ex.Xedge,ex.nodeMap,ex.edgeMap,ex.edgeStruct);
	yMAP = decodeFunc(nodePot,edgePot,ex.edgeStruct,varargin{:});

	% Compute sufficient statistics
	ocrep = overcompletePairwise(yMAP,ex.nState,ex.edgeStruct);
	ss_mu = Fx * ocrep;
	
	% Difference of sufficient statistics
	ssDiff = ss_mu - ss_y;
	
	% Objective
	loss = w'*ssDiff / (nEx*ex.nNode);
	f = f + loss;
	
	% Gradient
	if nargout == 2
		gradW = gradW + ssDiff / (nEx*ex.nNode);
	end
	
end

if nargout == 2
	g = gradW;
end


