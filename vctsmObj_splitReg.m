function [f, g] = vctsmObj_splitReg(x, examples, C, inferFunc, varargin)

% Outputs the objective value and gradient of the VCTSM learning objective
% using the dual of loss-augmented inference to make the objective a
% minimization.
%
% x : current point in optimization
% examples : nEx x 1 cell array of examples, each containing:
%	oc : full overcomplete vector representation of Y
%		 (including high-order terms)
%	ocLocalScope : number of local terms in oc
%	Aeq : nCon x length(oc) constraint A matrix
%	beq : nCon x 1 constraint b vector
%	Fx : nParam x length(oc) feature map
%	suffStat : nParam x 1 vector of sufficient statistics (i.e., Fx * oc)
% C : regularization constant or vector
% inferFunc : inference function
% varargin : optional arguments (required by minFunc)

nEx = length(examples);
nParam = max(examples{1}.edgeMap(:));

% parse current position
w = x(1:nParam);
logKappa = x(nParam+1);
kappa = exp(logKappa);

% init outputs
f = 0.5*(C.*w)'*w + 0.5*C/kappa^2;
if nargout == 2
	gradW = (C .* w);
	gradLogKappa = -C / kappa^2;
end

% main loop
for i = 1:nEx
	
	% Grab ith example.
	ex = examples{i};
	Fx = ex.Fx;
	ss_y = ex.suffStat;
	Ynode = overcompleteRep(ex.Y,ex.nState,0)';
	
	% Loss-augmented (approx) marginal inference
	[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex.Xnode,ex.Xedge,ex.nodeMap,ex.edgeMap,ex.edgeStruct);
	nodePot = nodePot .* exp(1 - Ynode);
	[nodeBel,edgeBel,logZ] = UGM_Infer_CBP(kappa,nodePot,edgePot,ex.edgeStruct,inferFunc,varargin{:});
	
	% Compute sufficient statistics
	mu = [reshape(nodeBel',[],1) ; edgeBel(:)];
	ss_mu = Fx*mu;
	
	% Compute pseudo-entropy of pseudo-marginals using the identity:
	% logZ = U - H, where H is entropy and U = w' * Fx * mu
	U = w' * ss_mu;
	H = logZ - U;
	
	% Scale pseudo-entropy by kappa (or exp(logKappa))
	H = H * kappa;

	% objective
	L1 = norm(Ynode(:)-nodeBel(:), 1);
	loss = U - w'*ss_y + H + L1;
	f = f + loss;
	
	% gradient
	if nargout == 2
		gradW = gradW + ss_mu - ss_y;
		gradLogKappa = gradLogKappa + H;
	end
	
end

if nargout == 2
	g = [gradW; gradLogKappa];
end


