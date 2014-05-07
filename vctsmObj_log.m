function [f, g] = vctsmObj_log(x, examples, C1, C2, inferFunc, varargin)
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
nParam = max(examples{1}.edgeMap(:));

% Parse current position
w = x(1:nParam);
logKappa = x(nParam+1);
kappa = exp(logKappa);

% kappa must be positive
if kappa <= 0
	err = MException('vctsmObj:BadInput',...
			sprintf('kappa must be strictly positive; log(kappa)=%f, kappa=%f',logKappa,kappa));
	throw(err);
end

%% Regularizer

% Convex upper bound regularizer using Young's inequality
f = 0.5*C1 * (C2*(w'*w) + 1/(C2*kappa^2));
if nargout == 2
	gradW = C1 * C2 * w;
	gradLogKappa = -C1 / (C2*kappa^2);
end

% Non-convex regularizer
% f = 0.5 * C1 * (w'*w) / kappa^2;
% if nargout == 2
% 	gradW = C1 * w / kappa^2;
% 	gradLogKappa = -C1 * (w'*w) / kappa^2;
% end

%% Main loop

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
	
	% Gradient
	if nargout == 2
		gradW = gradW + ssDiff / (nEx*ex.nNode);
		gradLogKappa = gradLogKappa + (kappa*H) / (nEx*ex.nNode);
	end
	
end

if nargout == 2
	g = [gradW ; gradLogKappa];
end


