function [f, g] = sctsmObj(x, examples, C, inferFunc, kappa, varargin)
%
% Outputs the objective value and gradient of the SCTSM learning objective
% (i.e., VCTSM with a fixed kappa).
%
% x : current point in optimization
% examples : nEx x 1 cell array of examples, each containing:
%	Fx : nParam x length(oc) feature map
%	suffStat : nParam x 1 vector of sufficient statistics (i.e., Fx * oc)
%	Ynode : nState x nNode overcomplete matrix representation of labels
% C : regularization constant or vector
% inferFunc : inference function
% kappa : predefined modulus of convexity
% varargin : optional arguments (required by minFunc)

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
	Ynode = ex.Ynode'; % assumes Ynode is (nState x nNode)
	
	% Loss-augmented (approx) marginal inference
	[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex.Xnode,ex.Xedge,ex.nodeMap,ex.edgeMap,ex.edgeStruct);
	[nodeBel,edgeBel,~,H] = UGM_Infer_ConvexBP(kappa,nodePot.*exp(1-Ynode),edgePot,ex.edgeStruct,inferFunc,varargin{:});

	% Compute sufficient statistics
	ss_mu = Fx * [reshape(nodeBel',[],1) ; edgeBel(:)];
	
	% Difference of sufficient statistics
	ssDiff = ss_mu - ss_y;
	
	% Objective
	% Note: -\Psi = H
	L1 = norm(Ynode(:)-nodeBel(:), 1);
	loss = (w'*ssDiff + kappa*H + 0.5*L1) / (nEx*ex.nNode);
	f = f + loss;

% 	if f < 0
% 		f = inf;
% 	end
	
	% Gradient
	if nargout == 2
		gradW = gradW + ssDiff / (nEx*ex.nNode);
	end
	
end

if nargout == 2
	g = gradW;
end


