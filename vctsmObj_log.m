function [f, g] = vctsmObj_log(x, examples, C, inferFunc, varargin)
%
% Outputs the objective value and gradient of the VCTSM learning objective.
%
% x : current point in optimization
% examples : nEx x 1 cell array of examples, each containing:
%	Fx : nParam x length(oc) feature map
%	suffStat : nParam x 1 vector of sufficient statistics (i.e., Fx * oc)
%	Ynode : nState x nNode overcomplete matrix representation of labels
% C : regularization constant
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

% Convex regularizer
f = 0.5 * C * (w'*w) / kappa;
if nargout == 2
	gradW = C * w / kappa;
	gradLogKappa = -0.5 * C * (w'*w) / kappa;
end

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


