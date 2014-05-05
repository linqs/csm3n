function [f, g] = vctsmObj(x, examples, C1, C2, inferFunc, varargin)
%
% Outputs the objective value and gradient of the VCTSM learning objective.
%
% x : current point in optimization
% examples : nEx x 1 cell array of examples, each containing:
%	Fx : nParam x length(oc) feature map
%	suffStat : nParam x 1 vector of sufficient statistics (i.e., Fx * oc)
%	Ynode : nState x nNode overcomplete matrix representation of labels
% C1 : regularization constant for reg/loss tradeoff
% C2 : regularization constant for weights/convexity tradeoff
% inferFunc : inference function
% varargin : optional arguments (required by minFunc)

nEx = length(examples);

% Parse current position
w = x(1:end-1);
kappa = x(end);

% kappa must be positive
if kappa <= 0
	err = MException('vctsmObj:BadInput',...
			sprintf('kappa must be strictly positive; kappa=%f',kappa));
	throw(err);
end

% Init outputs
% Convex upper bound regularizer using Young's inequality
f = 0.5*C1 * (C2*(w'*w) + 1/(C2*kappa^2));
if nargout == 2
	gradW = C1 * C2 * w;
	gradKappa = -C1 / (C2*kappa^3);
end
% Non-convex regularizer
% f = 0.5 * C1 * (w'*w) / kappa^2;
% if nargout == 2
% 	gradW = C1 * w / kappa^2;
% 	gradKappa = -C1 * (w'*w) / (C2*kappa^3);
% end

% Main loop
for i = 1:nEx
	
	% Grab ith example.
	ex = examples{i};
	Fx = ex.Fx;
	ss_y = ex.suffStat;
	Ynode = ex.Ynode'; % assumes Ynode is (nState x nNode)
	
	% Loss-augmented (approx) marginal inference
	[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex.Xnode,ex.Xedge,ex.nodeMap,ex.edgeMap,ex.edgeStruct);
	[nodeBel,edgeBel,logZ] = UGM_Infer_ConvexBP(kappa,nodePot.*exp(1-Ynode),edgePot,ex.edgeStruct,inferFunc,varargin{:});
	
	% Compute sufficient statistics
	mu = [reshape(nodeBel',[],1) ; edgeBel(:)];
	ss_mu = Fx*mu;
	
	% Compute pseudo-entropy of pseudo-marginals using the identity:
	% logZ = U + H, where H is entropy and U = w' * Fx * mu
	U = w' * ss_mu;
	H = logZ - U;
	
	% Objective
	% Note: -\Psi = H
	L1 = norm(Ynode(:)-nodeBel(:), 1);
	loss = U - w'*ss_y + kappa*H + L1;
	f = f + loss / (nEx*ex.nNode);
	
	% Gradient
	if nargout == 2
		gradW = gradW + (ss_mu-ss_y) / (nEx*ex.nNode);
		gradKappa = gradKappa + H / (nEx*ex.nNode);
	end
	
end

if nargout == 2
	g = [gradW; gradKappa];
end


